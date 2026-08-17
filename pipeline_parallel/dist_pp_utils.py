from .dist_gpipe_pipeline_async import GpipeAsync


def get_pp_module(args, vocab_size, num_classes, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, vocab_size, num_classes, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
