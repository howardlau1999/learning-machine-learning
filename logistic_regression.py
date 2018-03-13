import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression():
    def __init__(self):
        pass
    # Fit a logistic regression line with data X and y
    # Returns cost history
    def fit(self, X, y, learning_rate = 0.001, steps = 100000):
        # X is original data, so we should add ones
        # Rows are features, cols are samples
        costs = []
        original = X.copy()
        self.m = X.shape[1]
        X = np.vstack((np.ones((1, self.m)), X))
        self.n = X.shape[0]
        self.parameters = np.zeros((1, self.n))
        # Gradient Descent (very slow)
        # However, if we just increase the learning rate by 0.0001
        # The loss will be extremely unstable
        # So it is better to use Newton's method to find the best parameters
        """
        for step in range(steps):
            h = sigmoid(np.dot(self.parameters, X))
            gradients = np.dot((h - y), X.T) / self.m
            self.parameters -= learning_rate * gradients
            loss = self.loss(X, y)
            costs.append(loss[0][0])
        """

        # Newton's method, faster, but more computationally expensive
        for step in range(steps):
            h = sigmoid(np.dot(self.parameters, X))
            # First derivative
            gradients = np.dot((h - y), X.T) / self.m
            h_matrix = np.zeros((self.n, self.n))
            # Second derivative
            # Didn't figure out the vectorized version, TODO
            for i in range(self.m):
                sample = X[:, i]
                sample = sample.reshape(self.n, 1)
                h_matrix += np.dot(sample, sample.T) * h[0][i] * (1 - h[0][i]) 
            self.parameters -= np.dot(gradients, np.linalg.pinv(h_matrix))
            loss = self.loss(X, y)
            costs.append(loss[0][0])
        X = original
        print(self.parameters, costs[-1])
        return costs

    def predict(self, X):
        original = X.copy()
        X = np.vstack((np.ones((1, self.m)), X))
        prediction = sigmoid(np.dot(self.parameters, X)) > 0.5
        X = original
        return prediction

    def loss(self, X, y):
        g = sigmoid(np.dot(self.parameters, X))
        # Overflow
        J = -(np.dot(np.log(g), y.T) + np.dot(np.log(1 - g), (1 - y).T)) / self.m
        return J

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def load_data(filename='ex2data1.txt', delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    X = data[:, :-1]
    X = X.T
    m = X.shape[1]
    y = data[:, -1]
    y = y.reshape((1, m))

    return (X, y)

def main():
    X, y = load_data()
    lr = LogisticRegression()
    history = lr.fit(X, y, 0.001, 1000)
    print('Train Accuracy: %.2f%%' % ((np.sum(lr.predict(X) == y)) / len(y)))
    plt.plot(history)
    plt.show()
if __name__ == '__main__':
    main()