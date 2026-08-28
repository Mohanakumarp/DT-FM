import torch


class SyntheticSequenceDataset(torch.utils.data.Dataset):
    """Small deterministic dataset for device and distributed smoke tests."""

    def __init__(self, sample_count, seq_length, vocab_size):
        if sample_count <= 0:
            raise ValueError('synthetic sample count must be positive')
        if vocab_size < 4:
            raise ValueError('synthetic vocab size must be at least 4')
        self.sample_count = sample_count
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.tokens = torch.arange(seq_length, dtype=torch.long)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        text = (self.tokens + index * 17) % self.vocab_size
        return {
            'text': text.clone(),
            'label': torch.tensor(index % 2, dtype=torch.long),
        }


def get_synthetic_train_data_loader(args):
    if args.num_epochs > 0:
        requested_steps = args.num_epochs * args.steps_per_epoch
    else:
        requested_steps = args.num_iters
    minimum_samples = max(args.batch_size * requested_steps, args.batch_size)
    sample_count = max(args.synthetic_samples, minimum_samples)
    dataset = SyntheticSequenceDataset(
        sample_count=sample_count,
        seq_length=args.seq_length,
        vocab_size=args.synthetic_vocab_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=getattr(args, 'pin_memory', False),
    )
