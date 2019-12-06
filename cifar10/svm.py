import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, input_size=3072, output_size=10):
        self.W = np.random.uniform(-0.005, 0.005, (input_size, output_size))

    def fit(self, train_images, train_labels, batch_size, learning_rate, epochs, reg, delta):
        for epoch in tqdm(range(epochs)):
            train_idx = np.random.choice(len(train_images), len(train_images), replace=False)
            batches = int(np.ceil(len(train_labels) / batch_size))
            for batch in range(batches):
                begin = batch * batch_size
                end = (batch + 1) * batch_size
                X = train_images[train_idx[begin:end]]
                y = train_labels[train_idx[begin:end]]
                m = X.shape[0]
                scores = self.predict(X)
                correct_scores = scores[range(m), y]
                margins = scores - correct_scores[:, np.newaxis] + delta
                margins[margins < 0] = 0
                margins[range(m), y] = 0

                loss = np.mean(margins) + .5 * reg * np.sum(self.W ** 2)

                ground_truths = np.zeros(margins.shape)
                ground_truths[margins > 0] = 1
                ground_truths[range(m), y] -= np.sum(ground_truths, axis=1)  # batch_size x C
                self.W -= learning_rate * (X.T @ ground_truths) / m + reg * self.W

    def predict(self, images):
        scores = images @ self.W
        return scores
