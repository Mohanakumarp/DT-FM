from comm.comm_utils import *


def total_train_steps(args):
    epochs = int(getattr(args, 'num_epochs', 0) or 0)
    spe = int(getattr(args, 'steps_per_epoch', 0) or 0)
    if epochs > 0:
        if spe <= 0:
            raise ValueError('--num-epochs requires --steps-per-epoch (QQP is ~364k samples/epoch)')
        return epochs * spe
    return int(args.num_iters)


def _batch_labels(args, data, device):
    if args.task == 'SeqClassification':
        return data['label'].to(device)
    elif args.task == 'Seq2SeqClassification':
        return data['text'].to(device)
    else:
        print("Not supported task!")
        assert False


def _iter_seconds(stats):
    if isinstance(stats, dict):
        return float(stats.get('iter_s', 0.0))
    return float(stats)


def _print_finished_iters(args, total_time, last_iter_time, completed):
    if completed > 1:
        averaged_time = total_time / (completed - 1)
        print("Finished running ", completed,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
    else:
        print("Finished running ", completed,
              " iterations, run time:", last_iter_time)


def _maybe_record(metrics, stats):
    if metrics is not None:
        metrics.record_iter(stats if isinstance(stats, dict) else {'iter_s': float(stats)})


def distributed_train_foo_iter(args, pipeline, device, train_data_loader, metrics=None):
    pp_rank = get_pipeline_parallel_rank()
    is_first = (pp_rank == 0)
    is_last = (pp_rank == args.pipeline_group_size - 1)
    steps_budget = total_train_steps(args)
    epochs = int(getattr(args, 'num_epochs', 0) or 0)
    spe = int(getattr(args, 'steps_per_epoch', 0) or 0)
    if epochs <= 0:
        epochs = 1
        spe = steps_budget

    print("Training budget: epochs=%d steps_per_epoch=%d total_steps=%d"
          % (epochs if getattr(args, 'num_epochs', 0) else 0, spe, steps_budget))

    if metrics is not None:
        metrics.mark('train_start')

    # A 1-stage pipeline is both first and last: it must consume token ids
    # and QQP labels on the same rank. Multi-stage first/last behavior is unchanged.
    completed = 0
    total_time = 0.0
    last_iter_time = None

    def _one_step(input_ids, labels):
        nonlocal completed, total_time, last_iter_time
        stats = pipeline.sgd_iter(input_ids, labels)
        last_iter_time = _iter_seconds(stats)
        _maybe_record(metrics, stats)
        if metrics is not None and getattr(pipeline, 'last_loss', None) is not None:
            metrics.record_loss(pipeline.last_loss)
        if completed > 0:
            total_time += last_iter_time
        completed += 1
        if getattr(args, 'rebalance_every', 0):
            from scheduler.rebalance import maybe_rebalance
            maybe_rebalance(args, pipeline, completed, steps_budget, metrics)

    if is_first and is_last:
        for epoch in range(epochs):
            print("==== Epoch", epoch, "/", epochs)
            n = 0
            for data in train_data_loader:
                _one_step(data['text'].to(device), _batch_labels(args, data, device))
                n += 1
                if n >= spe or completed >= steps_budget:
                    break
            if completed >= steps_budget:
                break
        _print_finished_iters(args, total_time, last_iter_time, completed)
    elif is_first:
        for epoch in range(epochs):
            print("==== Epoch", epoch, "/", epochs)
            n = 0
            for data in train_data_loader:
                _one_step(data['text'].to(device), None)
                n += 1
                if n >= spe or completed >= steps_budget:
                    break
            if completed >= steps_budget:
                break
        _print_finished_iters(args, total_time, last_iter_time, completed)
    elif is_last:
        for epoch in range(epochs):
            print("==== Epoch", epoch, "/", epochs)
            n = 0
            for data in train_data_loader:
                _one_step(None, _batch_labels(args, data, device))
                n += 1
                if n >= spe or completed >= steps_budget:
                    break
            if completed >= steps_budget:
                break
    else:
        for epoch in range(epochs):
            print("==== Epoch", epoch, "/", epochs)
            for _ in range(spe):
                _one_step(None, None)
                if completed >= steps_budget:
                    break
            if completed >= steps_budget:
                break

    if metrics is not None:
        metrics.mark('train_end')
