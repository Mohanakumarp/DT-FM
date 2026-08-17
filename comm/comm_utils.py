from .nccl_backend import *

_DATA_PARALLEL_COMM = None
_DATA_PARALLEL_RANK = None
_DATA_PARALLEL_WORLD_SIZE = None

_PIPELINE_PARALLEL_COMM = None
_PIPELINE_PARALLEL_RANK = None
_PIPELINE_PARALLEL_WORLD_SIZE = None


class GlooTensorCommunicator:
    """
    CPU-staged tensor send/recv over the existing Gloo process group.

    This is for same-machine / WSL multi-process tests when CuPy NCCL cannot
    initialize. Real multi-GPU DT-FM runs should keep using NCCLCommunicator.
    """

    def __init__(self, comm_rank, comm_group_size, comm_name, world_rank_base=0):
        self.comm_rank = comm_rank
        self.comm_group_size = comm_group_size
        self.comm_name = comm_name
        self.world_rank_base = world_rank_base
        self.dist_store = dist.distributed_c10d._get_default_store()
        print("Initialize GlooTensorCommunicator: <", comm_name, ">; rank:", comm_rank,
              "; world_rank_base:", world_rank_base)

    def _world_rank(self, group_rank):
        return self.world_rank_base + group_rank

    @staticmethod
    def barrier():
        dist.barrier()

    def store_set(self, key, value):
        self.dist_store.set(key, value)

    def store_get(self, key):
        return self.dist_store.get(key)

    def send(self, tensor, dst, stream=None):
        if tensor is None:
            raise ValueError("GlooTensorCommunicator.send got tensor=None")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        payload = tensor.detach().contiguous()
        if payload.is_cuda:
            payload = payload.cpu()
        dist.send(payload, dst=self._world_rank(dst))

    def recv(self, tensor, src, stream=None):
        if tensor is None:
            raise ValueError("GlooTensorCommunicator.recv got tensor=None")
        tmp = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu')
        dist.recv(tmp, src=self._world_rank(src))
        # NCCL writes through data_ptr(); copy_ into a requires_grad leaf
        # is rejected by autograd, so fill storage without tracking.
        with torch.no_grad():
            tensor.copy_(tmp)
        if tensor.is_cuda:
            torch.cuda.synchronize()

    def broadcast(self, tensor, src, stream=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        payload = tensor.detach().contiguous()
        was_cuda = payload.is_cuda
        if was_cuda:
            payload = payload.cpu()
        dist.broadcast(payload, src=self._world_rank(src))
        if was_cuda:
            with torch.no_grad():
                tensor.copy_(payload)

    def all_reduce(self, tensor, stream=None, op=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        payload = tensor.detach().contiguous()
        was_cuda = payload.is_cuda
        if was_cuda:
            payload = payload.cpu()
        dist.all_reduce(payload)
        with torch.no_grad():
            tensor.copy_(payload)


class SingleGPUCommunicator:
    """
    No-op communicator used only for a world-size-1 smoke test.

    With one GPU there is no inter-GPU communication required.
    This is NOT used for the actual multi-GPU DT-FM experiment.
    """

    def __init__(self, rank=0, world_size=1):
        self.comm_rank = rank
        self.comm_group_size = world_size

    @staticmethod
    def barrier():
        pass

    def store_set(self, key, value):
        pass

    def store_get(self, key):
        return None

    def send(self, tensor, dst, stream=None):
        # No communication needed for one GPU.
        pass

    def recv(self, tensor, src, stream=None):
        # No communication needed for one GPU.
        pass

    def broadcast(self, tensor, src, stream=None):
        # Source is this same process.
        pass

    def reduce(self, tensor, dst, stream=None, op=None):
        # Reduction over one process is the identity.
        pass

    def all_reduce(self, tensor, stream=None, op=None):
        # All-reduce over one process is the identity.
        pass

    def scatter(self, tensor, scatter_list, src, stream=None):
        # One process: copy the only input if necessary.
        if scatter_list:
            tensor.copy_(scatter_list[0])

    def gather(self, tensor, gather_list, dst, stream=None):
        # One process: copy into the only output if necessary.
        if gather_list:
            gather_list[0].copy_(tensor)

    def all_to_all(self, output_tensor_list, input_tensor_list, stream=None):
        # One process: identity.
        if output_tensor_list and input_tensor_list:
            output_tensor_list[0].copy_(input_tensor_list[0])

    def all_gather(self, tensor, output_tensor_list, stream=None):
        # One process: identity.
        if output_tensor_list:
            output_tensor_list[0].copy_(tensor)


def get_data_parallel_comm():
    assert _DATA_PARALLEL_COMM is not None
    return _DATA_PARALLEL_COMM


def get_data_parallel_rank():
    assert _DATA_PARALLEL_RANK is not None
    return _DATA_PARALLEL_RANK


def get_data_parallel_world_size():
    assert _DATA_PARALLEL_WORLD_SIZE is not None
    return _DATA_PARALLEL_WORLD_SIZE


def get_pipeline_parallel_comm():
    assert _PIPELINE_PARALLEL_COMM is not None
    return _PIPELINE_PARALLEL_COMM


def get_pipeline_parallel_rank():
    assert _PIPELINE_PARALLEL_RANK is not None
    return _PIPELINE_PARALLEL_RANK


def get_pipeline_parallel_world_size():
    assert _PIPELINE_PARALLEL_WORLD_SIZE is not None
    return _PIPELINE_PARALLEL_WORLD_SIZE


def _nccl_ok_consensus(local_ok, world_size):
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int64)
    gathered = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, flag)
    return all(int(t.item()) == 1 for t in gathered)


def _try_nccl_communicator(comm_rank, cuda_id, comm_group_size, comm_name):
    try:
        return NCCLCommunicator(comm_rank, cuda_id, comm_group_size, comm_name)
    except Exception as exc:
        print("NCCLCommunicator init failed for", comm_name,
              "rank", comm_rank, ":", repr(exc))
        return None


def _build_pipeline_comm(args, comm_rank, comm_group_size, comm_name, world_rank_base):
    tensor_comm = getattr(args, 'tensor_comm', 'auto')
    if tensor_comm == 'gloo':
        return GlooTensorCommunicator(
            comm_rank, comm_group_size, comm_name, world_rank_base
        )

    if tensor_comm not in ('auto', 'nccl'):
        raise ValueError("Unknown --tensor-comm: " + str(tensor_comm))

    nccl_comm = _try_nccl_communicator(
        comm_rank, args.cuda_id, comm_group_size, comm_name
    )
    nccl_ok = _nccl_ok_consensus(nccl_comm is not None, args.world_size)
    if nccl_ok:
        return nccl_comm

    if tensor_comm == 'nccl':
        raise RuntimeError(
            "NCCL tensor communication was requested but communicator init failed"
        )

    print("Falling back to Gloo tensor communication for", comm_name)
    return GlooTensorCommunicator(
        comm_rank, comm_group_size, comm_name, world_rank_base
    )


def init_communicators(args):
    default_init(args)

    assert args.world_size == (
        args.data_group_size * args.pipeline_group_size
    )

    global _DATA_PARALLEL_COMM
    global _PIPELINE_PARALLEL_COMM
    global _DATA_PARALLEL_RANK
    global _PIPELINE_PARALLEL_RANK
    global _DATA_PARALLEL_WORLD_SIZE
    global _PIPELINE_PARALLEL_WORLD_SIZE

    # ---------------------------------------------------------
    # Pipeline parallel group
    # ---------------------------------------------------------

    _PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size
    _PIPELINE_PARALLEL_RANK = args.rank % args.pipeline_group_size

    pipeline_group_name = "pipeline_group_" + str(
        args.rank // args.pipeline_group_size
    )
    pipeline_world_rank_base = (
        (args.rank // args.pipeline_group_size) * args.pipeline_group_size
    )

    if args.world_size == 1:
        # Single-GPU smoke test:
        # replace NCCL with a local identity communicator.
        _PIPELINE_PARALLEL_COMM = SingleGPUCommunicator(
            rank=0,
            world_size=1
        )
    else:
        _PIPELINE_PARALLEL_COMM = _build_pipeline_comm(
            args,
            _PIPELINE_PARALLEL_RANK,
            args.pipeline_group_size,
            pipeline_group_name,
            pipeline_world_rank_base
        )

    # ---------------------------------------------------------
    # Data parallel group
    # ---------------------------------------------------------

    _DATA_PARALLEL_WORLD_SIZE = args.data_group_size
    _DATA_PARALLEL_RANK = (
        args.rank // args.pipeline_group_size
    )

    if args.data_group_size == 1:
        _DATA_PARALLEL_COMM = SingleGPUCommunicator(
            rank=0,
            world_size=1
        )

    elif args.world_size > 1:
        _DATA_PARALLEL_COMM = NCCLCommunicator(
            _DATA_PARALLEL_RANK,
            args.cuda_id,
            args.data_group_size,
            "data_group_" + str(
                args.rank % args.pipeline_group_size
            )
        )

    else:
        _DATA_PARALLEL_COMM = None
