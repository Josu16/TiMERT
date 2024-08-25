import numpy as np

class NumpyDataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        for start_idx in range(0, len(self.data), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.data))
            batch_indices = self.indices[start_idx:end_idx]
            batch = self.data[batch_indices]
            yield batch, None

    def __len__(self):
        return len(self.data) // self.batch_size
