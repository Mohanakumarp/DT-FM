from comm.comm_utils import *


def _batch_labels(args, data, device):
    if args.task == 'SeqClassification':
        return data['label'].to(device)
    elif args.task == 'Seq2SeqClassification':
        return data['text'].to(device)
    else:
        print("Not supported task!")
        assert False


def _print_finished_iters(args, total_time, last_iter_time):
    if args.num_iters > 1:
        averaged_time = total_time / (args.num_iters - 1)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
    else:
        print("Finished running ", args.num_iters,
              " iterations, run time:", last_iter_time)


def distributed_train_foo_iter(args, pipeline, device, train_data_loader):
    pp_rank = get_pipeline_parallel_rank()
    is_first = (pp_rank == 0)
    is_last = (pp_rank == args.pipeline_group_size - 1)

    # A 1-stage pipeline is both first and last: it must consume token ids
    # and QQP labels on the same rank. Multi-stage first/last behavior is unchanged.
    if is_first and is_last:
        total_time = 0
        last_iter_time = None
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            labels = _batch_labels(args, data, device)
            current_iter_time = pipeline.sgd_iter(input_ids, labels)
            last_iter_time = current_iter_time
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters - 1:
                break
        _print_finished_iters(args, total_time, last_iter_time)
    elif is_first:
        total_time = 0
        last_iter_time = None
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.sgd_iter(input_ids, None)
            last_iter_time = current_iter_time
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters - 1:
                break
        _print_finished_iters(args, total_time, last_iter_time)
    elif is_last:
        for i, data in enumerate(train_data_loader):
            labels = _batch_labels(args, data, device)
            pipeline.sgd_iter(None, labels)
            if i >= args.num_iters - 1:
                break
    else:
        i = 0
        while True:
            pipeline.sgd_iter(None, None)
            i += 1
            if i >= args.num_iters:
                break
