import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, seq_len, num_samples=20000, vocab_size=70000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]

def get_dummy_dataset(seq_len, batch_size):
    dataset = DummyDataset(seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.vocab_size
