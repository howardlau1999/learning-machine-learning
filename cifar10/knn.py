import numpy as np

class KNN:
    def __init__(self, k):
        self.train_images = np.array([])
        self.train_labels = np.array([])
        self.k = k

    def fit(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def predict(self, image):
        distances = np.linalg.norm(self.train_images - image, axis=1)
        neighbours = self.train_labels[np.argsort(distances)[:self.k]]
        return np.argmax(np.bincount(neighbours))
