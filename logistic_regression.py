import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
class LogisticRegression():
    def __init__(self):
        pass
    # Fit a logistic regression line with data X and y
    # Returns cost history
    def fit(self, X, y, learning_rate = 0.001, steps = 1000, lambd = 0.01, epsilon = 1e-6, retrain = True, optimization='BFGS'):
        # X is original data, so we should add ones
        # Rows are features, cols are samples
        costs = []
        original = X.copy()
        self.m = X.shape[1]
        X = np.vstack((np.ones((1, self.m)), X))
        self.n = X.shape[0]

        if retrain or not hasattr(self, 'parameters'):
            self.parameters = np.random.rand(1, self.n) / self.n / 1000
        # Gradient Descent (very slow)
        # However, if we just increase the learning rate by 0.0001
        # The loss will be extremely unstable
        # So it is better to use Newton's method to find the best parameters
        if optimization == 'gradient_descent':
            for step in range(steps):
                gradients = self.gradients(X, y, lambd = lambd)
                self.parameters -= learning_rate * gradients
                loss = self.loss(X, y, lambd = lambd)
                costs.append(loss[0][0])
        elif optimization.startswith('newton'):
        # Newton's method, faster, but more computationally expensive
            step = 0
            gradients = self.gradients(X, y, lambd = lambd)
            while step < steps and np.linalg.norm(gradients) > epsilon:
                # First derivative
                h = sigmoid(np.dot(self.parameters, X))

                gradients = self.gradients(X, y, lambd = lambd)
                H = np.zeros((self.n, self.n))
                # Second derivative
                # Didn't figure out the vectorized version, TODO
                for i in range(self.m):
                    sample = X[:, i]
                    sample = sample.reshape(self.n, 1)
                    # Need to correct the equation here
                    H += (np.dot(sample, sample.T) * h[0][i] * (1 - h[0][i]))
                d_k = -np.dot(gradients, np.linalg.pinv(H))
                step_length = 1
                if optimization.endswith('_damp'):
                    step_length = self.find_step_length(X, y, d_k, lambd = lambd)
                self.parameters += step_length * d_k
                step = step + 1
                loss = self.loss(X, y, lambd = lambd)
                costs.append(loss[0][0])
        # Quasi-newton method, no need to compute the second derivatives
        # But use first derivatives to approximate H^{-1}
        # BFGS method
        else:
            # Initializations
            if retrain or not hasattr(self, 'D_k'):
                self.D_k = np.eye(self.n)
                self.g_k = self.gradients(X, y, lambd = lambd)
                self.s_k = np.zeros((1, self.n))
                self.d_k = np.zeros((1, self.n))
                self.y_k = np.zeros((1, self.n))
            # Step is of no use here
            # Use epsilon to end
            
            step = 0
            while np.linalg.norm(self.g_k) >= epsilon and step < steps:
                self.d_k = -np.dot(self.g_k, self.D_k)
                # Find alpha
                step_length = self.find_step_length(X, y, self.d_k, lambd = lambd)
                # Update parameters with step_length

                self.s_k = step_length * self.d_k
                self.parameters += self.s_k
                old_g = self.g_k.copy()
                self.g_k = self.gradients(X, y, lambd = lambd)
                self.y_k = self.g_k - old_g
                ys = np.dot(self.y_k, self.s_k.T)
                ident = np.eye(self.n)
                mat_left = np.asmatrix((ident - np.dot(self.s_k.T, self.y_k) / ys))
                mat_right = np.asmatrix((ident - np.dot(self.y_k.T, self.s_k) / ys))
                self.D_k = mat_left * np.asmatrix(self.D_k) * mat_right + np.dot(self.s_k.T, self.s_k) / ys
                    
                step = step + 1
                loss = self.loss(X, y, lambd = lambd)
                costs.append(loss[0][0])

        X = original
        print(self.parameters, costs[-1])
        return costs

    def find_step_length(self, X, y, d_k, lambd = 0):
        # 1. Forward and backword method
        #    Find the range[a, b] we should search in
        #    It's important that we choose an appropriate initial value
        h = np.random.rand()
        t = 2
        a = 0
        b = 0
        alpha_1 = 0
        alpha_2 = alpha_1 + h
        alpha_3 = 0
        compute_guess = lambda alpha : self.loss(X, y, self.parameters + alpha * d_k, lambd = lambd)
        f1 = compute_guess(alpha_1)
        f2 = compute_guess(alpha_2)
        while True:
            # Step 2
            if f1 > f2:
                h = t * h
            else:
                h = -h
                alpha_1, alpha_2 = alpha_2, alpha_1
                f1, f2 = f2, f1
            alpha_3 = alpha_2 + h
            f3 = compute_guess(alpha_3)
            if f3 > f2:
                a = min(alpha_1, alpha_3)
                b = max(alpha_1, alpha_3)
                break
            else:
                alpha_1 = alpha_2
                alpha_2 = alpha_3
                f1 = f2
                f2 = f3
        # 2. 0.618 method, find the best alpha
        delta = 1e-6
        # Magic number
        a_k = a
        b_k = b
        step_length = 0
        lambda_k = a_k + 0.382 * (b_k - a_k)
        mu_k = a_k + 0.618 * (b_k - a_k)
        f_l = compute_guess(lambda_k)
        f_m = compute_guess(mu_k)
        while True:
            if f_l > f_m:
                if b_k - lambda_k <= delta:
                    return mu_k
                else:
                    a_k = lambda_k
                    lambda_k = mu_k
                    f_l = f_m
                    mu_k = a_k + 0.618 * (b_k - a_k)
                    f_m = compute_guess(mu_k)
            else:
                if mu_k - a_k <= delta:
                    return lambda_k
                else:
                    b_k = mu_k
                    mu_k = lambda_k
                    f_m = f_l
                    lambda_k = a_k + 0.382 * (b_k - a_k)
                    f_l = compute_guess(lambda_k)


    def predict(self, X):
        original = X.copy()
        m = X.shape[1]
        X = np.vstack((np.ones((1, m)), X))
        prediction = sigmoid(np.dot(self.parameters, X)) > 0.5
        X = original
        return prediction

    def loss(self, X, y, parameters = None, lambd = 0):
        if parameters is None:
            parameters = self.parameters
        g = sigmoid(np.dot(parameters, X))
        # Aviod divided by zeros
        epsilon = 1e-8
        J = -(np.dot(np.log(g + epsilon), y.T) + np.dot(np.log(1 - g + epsilon), (1 - y).T)) / self.m
        J += lambd / (2 * self.m) * np.linalg.norm(parameters[0, 1:]) ** 2
        return J

    def gradients(self, X, y, parameters = None, lambd = 0):
        if parameters is None:
            parameters = self.parameters
        gradients = np.dot((sigmoid(np.dot(self.parameters, X)) - y), X.T) / self.m
        gradients += np.hstack((0, lambd / self.m * gradients[0, 1:]))
        return gradients



def sigmoid(X):
    return .5 * (1 + np.tanh(.5 * X))
    # return 1 / (1 + np.exp(-X))

def decision_boundary(X, y, lr):
    plt.clf()
    delta = 0.5
    min_v = 20
    max_v = 110
    x1 = np.arange(min_v, max_v, delta)
    x2 = np.arange(min_v, max_v, delta)
    PX, PY = np.meshgrid(x1, x2)
    Z = lr.predict(np.c_[PX.ravel(), PY.ravel(), PX.ravel() ** 2, PY.ravel() ** 2].T)
    points = int((max_v - min_v) / delta)
    Z = Z.reshape(points, points)
    plt.contourf(PX, PY, Z, 8, alpha=.75, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)

def load_data(filename='ex2data1.txt', delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    X = data[:, :-1]
    X = X.T
    m = X.shape[1]
    y = data[:, -1]
    y = y.reshape((1, m))

    return (X, y)



def animate_fit(X, y, lr):
    fig, ax = plt.subplots()
    def animate_decision_boundary(i = 0):
        print(i)
        lr.fit(X, y, learning_rate=0.000001, steps=1,  retrain=False, optimization = 'BFGS')
        decision_boundary(X, y, lr)
        return fig,
    ani = animation.FuncAnimation(fig=fig,
                              func=animate_decision_boundary,
                              frames=500,
                              init_func=animate_decision_boundary,
                              interval=20,
                              blit=False)
    plt.show()

def main():
    X, y = load_data()
    lr = LogisticRegression()   
    X = np.vstack((X, X[0, :] ** 2, X[1, :] ** 2))
    # lr.fit(X, y, optimization='gradient_descent')
    # decision_boundary(X, y, lr)
    animate_fit(X, y, lr)
    plt.show()
if __name__ == '__main__':
    main()