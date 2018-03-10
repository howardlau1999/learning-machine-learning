import numpy as np

def feature_normalize(X):
    X_norm = X.copy()
    n_features = X_norm.shape[0]
    m = X_norm.shape[1]
    for i in range(n_features):
        samples = X_norm[i, :]
        mu = np.mean(samples)
        samples -= mu
        std = np.std(samples)
        samples /= std
        X_norm[i, :] = samples
    return X_norm

def loss(W, X, y):
    m = X.shape[1]
    y_hat = np.dot(W, X)
    return 0.5 * np.sum((y_hat - y) ** 2) / m

def linear_regression(X, y, learning_rate = 0.1, steps = 400):
    n = X.shape[0]
    m = X.shape[1]
    W = np.zeros((1, n))
    for step in range(steps):
        gradients = np.dot((np.dot(W, X) - y), X.T) / m
        W = W - learning_rate * gradients
    return W

def predict(W, x):
    return np.dot(W, x)

def normal_equation(X, y):
    X = X.T
    y = y.T
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y).T

def load_data(filename='ex1data2.txt', delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    X = data[:, :-1]
    X = X.T
    m = X.shape[1]
    y = data[:, -1]
    y = y.reshape((1, m))

    
    return (X, y)

def main():
    X, y = load_data('ex1data2.txt')
    m = X.shape[1]
    X_norm = np.vstack((np.ones((1, m)), feature_normalize(X)))
    X = np.vstack((np.ones((1, m)), X))
    W = linear_regression(X_norm, y)
    W_eqn = normal_equation(X, y)
    print('Weights using gradient descent:', W)
    print('Weights using normal equation:', W_eqn)
    
if __name__ == '__main__':
    main()