import numpy as np


class NaiveBayes:
    def __init__(self, vocab_size, alpha=0):
        assert 0 <= alpha <= 1, "alpha should be >= 0 and <= 1"
        self.vocab_size = vocab_size
        self.p_label = {}
        self.p_word_of_label = {}
        self.word_count_of_label = {}
        self.alpha = alpha
        self.labels = []

    def fit(self, X, Y):
        assert len(X) == len(Y), "Labels and samples does not match"
        self.labels = np.unique(Y)
        for label in self.labels:
            self.p_label[label] = np.sum(Y == label) / len(Y)
        for (x, y) in zip(X, Y):
            for word, freq in x.items():
                self.word_count_of_label[y] = self.word_count_of_label.get(y, 0) + freq
                self.p_word_of_label[word, y] = self.p_word_of_label.get((word, y), 0) + freq

        for k, v in self.p_word_of_label.items():
            word, label = k
            self.p_word_of_label[k] = (self.p_word_of_label[k] + self.alpha) / (
                    self.word_count_of_label[label] + self.alpha * (self.vocab_size + 1))
            self.p_word_of_label[k] = np.log(self.p_word_of_label[k])

    def predict(self, x):
        probs = {}

        for label in self.labels:
            probs[label] = self.p_label[label]
            for word, freq in x.items():
                probs[label] += self.p_word_of_label.get((word, label), np.log(
                    self.alpha / (self.word_count_of_label[label] + self.alpha * (self.vocab_size + 1)))) * freq
        return probs
