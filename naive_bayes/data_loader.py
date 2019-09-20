import os
import numpy as np

class DataLoader:
    def __init__(self, folder='DataPrepared'):
        self.trainX, self.trainY, self.testX, self.testY = [], [], [], []
        self.vocab_size = 0
        self.folder = folder

    @staticmethod
    def _load_features(fn):
        features = []
        vocab_size = 0
        with open(fn, 'r') as f:
            cur_sent_id = None
            cur_sent = None
            for line in f:
                sent_id, word_id, freq = [int(x) for x in line.split()]
                vocab_size = max(vocab_size, word_id)
                if cur_sent_id != sent_id:
                    if cur_sent is not None:
                        features.append(cur_sent)
                    cur_sent_id, cur_sent = sent_id, {}
                cur_sent[word_id] = freq
            features.append(cur_sent)
        return features, vocab_size

    @staticmethod
    def _load_labels(fn):
        labels = []
        with open(fn, 'r') as f:
            for line in f:
                labels.append(int(line))
        return labels

    def load_data(self, suffix=''):
        train_features_fn = os.path.join(self.folder, f'train-features{suffix}.txt')
        train_labels_fn = os.path.join(self.folder, f'train-labels{suffix}.txt')
        test_features_fn = os.path.join(self.folder, f'test-features.txt')
        test_labels_fn = os.path.join(self.folder, f'test-labels.txt')
        self.trainX, self.vocab_size = self._load_features(train_features_fn)
        self.testX, _ = self._load_features(test_features_fn)
        self.trainY, self.testY = self._load_labels(train_labels_fn), self._load_labels(test_labels_fn)
        self.trainX = np.array(self.trainX)
        self.trainY = np.array(self.trainY)
        self.testX = np.array(self.testX)
        self.testY = np.array(self.testY)

        assert len(self.trainX) == len(self.trainY)
        assert len(self.testX) == len(self.testY)

