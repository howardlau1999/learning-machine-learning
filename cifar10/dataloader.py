import pickle
from path import Path
import numpy as np


class DataLoader:
    def __init__(self):
        super().__init__()
        self.train_images = np.array([])
        self.train_labels = np.array([])
        self.test_images = np.array([])
        self.test_labels = np.array([])

    def load_data(self, batches_folder):
        self.train_images = np.array([])
        self.train_labels = np.array([])
        self.test_images = np.array([])
        self.test_labels = np.array([])

        for i in range(1, 6):
            with open(Path(batches_folder) / f"data_batch_{i}", "rb") as f:
                batch: dict = pickle.load(f, encoding='bytes')
                self.train_images = np.append(self.train_images, batch[b'data'])
                self.train_labels = np.append(self.train_labels, batch[b'labels'])
        self.train_images = self.train_images.reshape(-1, 3072)
        self.train_labels = self.train_labels.astype(dtype=np.int64)
        with open(Path(batches_folder) / f"test_batch", "rb") as f:
            batch: dict = pickle.load(f, encoding='bytes')
            self.test_images = np.array(batch[b'data'])
            self.test_labels = np.array(batch[b'labels'])

        self.test_images = self.test_images.reshape(-1, 3072)
        self.test_labels = self.test_labels.astype(dtype=np.int64)
