"""Transactional GPipe layer migration at completed FP32/SGD step boundaries."""
import copy
import hashlib
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from scheduler.dynamic_runtime import collect
from scheduler.pipeline_scheduler import apply_stage_assignment
from scheduler.resources import resource_snapshot


@contextmanager
def preserve_rng(device):
    cpu = torch.get_rng_state()
    api = torch.cuda if device.type == 'cuda' else torch.xpu if device.type == 'xpu' else None
    accelerator = api.get_rng_state(device) if api is not None else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if api is not None:
            api.set_rng_state(accelerator, device)


def stage_units(args, model):
    """Global layer identities, independent of local Sequential indices."""
    first = args.rank == 0
    children = list(model.model.children())
    layers = children[1:] if first else children
    if len(layers) != args.num_layers:
        raise ValueError('model layer count differs from the active assignment')
    units = {'layer.%d' % (args.layer_start + i): layer for i, layer in enumerate(layers)}
    if first:
        units['embedding'] = children[0]
    if args.rank == args.world_size - 1:
        units['head'] = model.task_layer
    return units


def unit_tensors(module, optimizer):
    result = {'weight/' + key: value.detach() for key, value in module.state_dict().items()}
    for name, parameter in module.named_parameters():
        state = optimizer.state.get(parameter, {})
        if set(state) - {'momentum_buffer'}:
            raise ValueError('migration supports SGD momentum_buffer state only')
        for key, value in state.items():
            if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != torch.float32:
                raise ValueError('unexpected SGD state tensor')
            result['momentum/' + name] = value.detach()
    if any(tensor.dtype != torch.float32 for tensor in result.values()):
        raise ValueError('migration requires FP32 state tensors')
    return result


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update((name + ':' + str(tuple(tensor.shape)) + ':' + str(tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_fingerprints(args, pipeline):
    return {name: tensor_digest(unit_tensors(module, pipeline.optimizer))
            for name, module in stage_units(args, pipeline.model).items()}


def owners(counts):
    return [rank for rank, count in enumerate(counts) for _ in range(count)]


def migrate(args, pipeline, counts):
    """Prepare, transfer, verify, then swap. Preparation failures leave old state.

    Rank loss or transport failure is fatal; this is not distributed crash
    recovery. Keep the caller at a fully drained optimizer-step boundary.
    """
    device = pipeline.device
    old_counts = [int(value) for value in args.stage_layers.split(',')]

    def validate():
        if (type(pipeline.optimizer) is not torch.optim.SGD or len(pipeline.optimizer.param_groups) != 1
                or pipeline.use_dp or pipeline.use_fp16 or pipeline.use_accelerator_streams
                or args.task != 'SeqClassification' or args.tensor_comm != 'gloo'):
            raise ValueError('migration requires single-group FP32 SGD with pipeline-only Gloo')
        if (len(counts) != args.world_size or any(type(n) is not int or n <= 0 for n in counts)
                or sum(counts) != sum(old_counts)):
            raise ValueError('migration must preserve all layers and one positive stage per rank')
        if args.num_layers != old_counts[args.rank] or args.layer_start != sum(old_counts[:args.rank]):
            raise ValueError('local layer range differs from the agreed active partition')
        stage_units(args, pipeline.model)
        options = {key: value for key, value in pipeline.optimizer.param_groups[0].items() if key != 'params'}
        return dict(old=old_counts, new=counts, options=options)
    agreements = collect(args, validate)
    if any(item != agreements[0] for item in agreements):
        raise ValueError('ranks disagree on migration assignments or SGD options')
    if counts == old_counts:
        return dict(status='unchanged', moved_layers=0)
    if args.world_size > 1:
        dist.barrier()
    # All previous gradients have been applied. Removing them does not alter
    # training and avoids retaining unnecessary storage during the transaction.
    pipeline.optimizer.zero_grad(set_to_none=True)
    old_owner, new_owner = owners(old_counts), owners(counts)
    moving = [index for index in range(sum(counts)) if old_owner[index] != new_owner[index]]
    units = stage_units(args, pipeline.model)

    def metadata():
        result = {}
        for index in moving:
            if old_owner[index] == args.rank:
                name = 'layer.%d' % index
                tensors = unit_tensors(units[name], pipeline.optimizer)
                result[name] = dict(shapes={key: list(t.shape) for key, t in tensors.items()},
                                    digest=tensor_digest(tensors),
                                    bytes=sum(t.numel() * t.element_size() for t in tensors.values()))
        return result
    pieces = collect(args, metadata)
    descriptions = {key: value for piece in pieces for key, value in piece.items()}

    def headroom():
        resource = resource_snapshot(args, device)
        incoming = sum(descriptions['layer.%d' % i]['bytes'] for i in moving if new_owner[i] == args.rank)
        outgoing = sum(descriptions['layer.%d' % i]['bytes'] for i in moving if old_owner[i] == args.rank)
        return dict(resources=resource, host_extra=3 * incoming + 2 * outgoing, device_extra=2 * incoming)
    budgets = collect(args, headroom)
    pools = {}
    for budget in budgets:
        resource = budget['resources']
        checks = [(('host', resource['host']), resource['host_memory']['available_bytes'], budget['host_extra'])]
        if resource['device_memory'] is not None:
            checks.append((('device', resource['host'], resource['device_pool']),
                           resource['device_memory']['available_bytes'], budget['device_extra']))
        for key, free, needed in checks:
            cap = max(0, int(free * args.scheduler_memory_fraction) - args.scheduler_reserve_mb * 1048576)
            previous = pools.setdefault(key, dict(cap=cap, needed=0))
            previous['cap'] = min(previous['cap'], cap)
            previous['needed'] += needed
    if any(pool['needed'] > pool['cap'] for pool in pools.values()):
        raise ValueError('insufficient temporary memory to preserve the old partition during migration')

    incoming_modules, transfers, incoming_momentum = {}, {}, {}
    with preserve_rng(device):
        def prepare():
            for index in moving:
                name = 'layer.%d' % index
                description = descriptions[name]
                if old_owner[index] == args.rank:
                    # CPU tensors can reference the frozen old stage. GPU state
                    # is staged through host memory for Gloo.
                    transfers[name] = {key: value.cpu().contiguous() for key, value in
                                       unit_tensors(units[name], pipeline.optimizer).items()}
                elif new_owner[index] == args.rank:
                    layer = pipeline.model._create_transformer_layer()
                    incoming_modules[name] = layer
                    weights = {'weight/' + key: value for key, value in layer.state_dict().items()}
                    expected_weights = {key: shape for key, shape in description['shapes'].items() if key.startswith('weight/')}
                    if {key: list(t.shape) for key, t in weights.items()} != expected_weights:
                        raise ValueError('incoming layer architecture differs from the source')
                    momentum = {}
                    parameters = dict(layer.named_parameters())
                    for key, shape in description['shapes'].items():
                        if key.startswith('momentum/'):
                            parameter_name = key[len('momentum/'):]
                            if parameter_name not in parameters or list(parameters[parameter_name].shape) != shape:
                                raise ValueError('incoming momentum does not match its parameter')
                            momentum[parameter_name] = torch.empty(shape, dtype=torch.float32)
                            weights[key] = momentum[parameter_name]
                    transfers[name] = weights
                    incoming_momentum[name] = momentum
        collect(args, prepare)
        # All source and destination storage exists before the first send. The
        # global order is identical everywhere, including non-adjacent moves.
        for index in moving:
            name = 'layer.%d' % index
            for key in sorted(descriptions[name]['shapes']):
                if old_owner[index] == args.rank:
                    dist.send(transfers[name][key], dst=new_owner[index])
                elif new_owner[index] == args.rank:
                    dist.recv(transfers[name][key], src=old_owner[index])

        candidate = {}
        def build_candidate():
            for name, module in incoming_modules.items():
                if tensor_digest(transfers[name]) != descriptions[name]['digest']:
                    raise ValueError('received layer state checksum mismatch')
                module.to(device)
            first = sum(counts[:args.rank])
            layers = [incoming_modules.get('layer.%d' % i, units.get('layer.%d' % i))
                      for i in range(first, first + counts[args.rank])]
            if any(layer is None for layer in layers):
                raise ValueError('missing layer in candidate stage')
            model = copy.copy(pipeline.model)
            model._modules = pipeline.model._modules.copy()
            model.model = nn.Sequential(*(([units['embedding']] if args.rank == 0 else []) + layers))
            model._num_layers = counts[args.rank]
            optimizer = torch.optim.SGD(model.parameters(), lr=agreements[0]['options']['lr'])
            optimizer.param_groups[0].update(agreements[0]['options'])
            optimizer.defaults = copy.deepcopy(pipeline.optimizer.defaults)
            for parameter in model.parameters():
                if parameter in pipeline.optimizer.state:
                    optimizer.state[parameter] = pipeline.optimizer.state[parameter]
            for name, momentum in incoming_momentum.items():
                for parameter_name, buffer in momentum.items():
                    parameter = dict(incoming_modules[name].named_parameters())[parameter_name]
                    optimizer.state[parameter] = {'momentum_buffer': buffer.to(device)}
            # Check after device conversion too, before peers vote to commit.
            for name, module in incoming_modules.items():
                if tensor_digest(unit_tensors(module, optimizer)) != descriptions[name]['digest']:
                    raise ValueError('candidate device state checksum mismatch')
            candidate.update(model=model, optimizer=optimizer)
        collect(args, build_candidate)
        # The collective above is the unanimous prepare vote. No fallible work
        # is performed between this point and swapping the active references.
        pipeline.model, pipeline.optimizer = candidate['model'], candidate['optimizer']
        args.stage_layers = ','.join(map(str, counts))
        apply_stage_assignment(args)
    if args.world_size > 1:
        dist.barrier()
    return dict(status='committed', moved_layers=len(moving),
                state_checksums={name: item['digest'] for name, item in descriptions.items()})
