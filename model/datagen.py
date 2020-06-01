import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size=16, shuffle=True, mode='train', augs=None):
        self.dim = data.shape[1]
        self.batch_size = batch_size
        assert mode in ['train', 'val']
        self.mode = mode
        self.labels = labels
        self.data = data
        self.n_channels = data.shape[2]
        self.shuffle = shuffle
        self.augs = augs
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(0, len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # Initialization
        x = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim, 11))

        # Generate data
        for i, idx in enumerate(indexes):
            x_, y_ = self.data[idx], self.labels[idx]

            if self.augs and self.mode == 'train':
                for aug, p in self.augs:
                    if np.random.rand() < p:
                        x_, y_ = aug(x_, y_)

            x[i] = x_
            y[i] = y_
        return x, y
