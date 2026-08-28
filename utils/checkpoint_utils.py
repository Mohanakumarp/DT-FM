import inspect

from torch.utils.checkpoint import checkpoint


_SUPPORTS_USE_REENTRANT = 'use_reentrant' in inspect.signature(checkpoint).parameters


def checkpoint_module(function, *args):
    if _SUPPORTS_USE_REENTRANT:
        return checkpoint(function, *args, use_reentrant=False)
    return checkpoint(function, *args)
