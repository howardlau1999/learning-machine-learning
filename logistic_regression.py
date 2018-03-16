import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Optimizer():
    def __init__(self, X, y, parameters, loss_function, gradients_function):
        self.X = X
        self.y = y
        self.parameters = parameters
        self.loss = loss_function
        self.gradients = gradients_function

class GradientDescentOptimizer(Optimizer):
    def __init__(self, X, y, parameters, loss_function, gradients_function, learning_rate = 0.001, lambd = 0.01):
        Optimizer.__init__(self, X, y, parameters, loss_function, gradients_function)
        self.learning_rate = learning_rate
        self.lambd = lambd

    def step(self):
        gradients = self.gradients(self.X, self.y, parameters=self.parameters, lambd=self.lambd)
        self.parameters -= self.learning_rate * gradients
        loss = self.loss(self.X, self.y, parameters=self.parameters, lambd=self.lambd)[0][0]
        return loss

class NewtonMethodOptimizer():
    def __init__(self, X, y, parameters, loss_function, gradients_function, second_derivatives_function, lambd=0.01, epsilon=1e-6):
        Optimizer.__init__(self, X, y, parameters, loss_function, gradients_function)
        self.second_derivatives = second_derivatives_function
        self.lambd = lambd
        self.epsilon = epsilon

    def step(self):
        n, m = self.X.shape
        gradients = self.gradients(self.X, self.y, parameters=self.parameters, lambd=self.lambd)
        H = self.second_derivatives(self.X, self.y, parameters=self.parameters, lambd=self.lambd)
        d_k = -np.dot(gradients, np.linalg.pinv(H))
        f = lambda alpha : self.loss(self.X, self.y, parameters=self.parameters + alpha * d_k, lambd=self.lambd)
        alpha = line_search(self.X, self.y, f)
        self.parameters += alpha * d_k
        loss = self.loss(self.X, self.y, parameters=self.parameters, lambd=self.lambd)[0][0]
        return loss

class BFGSOptimizer(Optimizer):
    def __init__(self, X, y, parameters, loss_function, gradients_function, lambd=0.01, epsilon=1e-6):
        Optimizer.__init__(self, X, y, parameters, loss_function, gradients_function)
        self.lambd = lambd
        self.epsilon = epsilon

        n, m = X.shape
        self.D_k = np.eye(n)
        self.g_k = self.gradients(self.X, self.y, parameters=self.parameters, lambd=self.lambd)
        self.s_k = np.zeros((1, n))
        self.d_k = np.zeros((1, n))
        self.y_k = np.zeros((1, n))
    def step(self):
        n, m = self.X.shape
        self.d_k = -np.dot(self.g_k, self.D_k)
        f = lambda alpha : self.loss(self.X, self.y, parameters=self.parameters + alpha * self.d_k, lambd=self.lambd)
        alpha = line_search(self.X, self.y, f)
        self.s_k = alpha * self.d_k
        self.parameters += self.s_k
        old_g = self.g_k.copy()
        self.g_k = self.gradients(self.X, self.y, parameters=self.parameters, lambd=self.lambd)
        self.y_k = self.g_k - old_g
        # Update inversed approximate Hessian matrix
        ys = np.dot(self.y_k, self.s_k.T)
        ident = np.eye(n)
        mat_left = np.asmatrix((ident - np.dot(self.s_k.T, self.y_k) / ys))
        mat_right = np.asmatrix((ident - np.dot(self.y_k.T, self.s_k) / ys))
        self.D_k = mat_left * np.asmatrix(self.D_k) * mat_right + np.dot(self.s_k.T, self.s_k) / ys
        loss = self.loss(self.X, self.y, parameters=self.parameters, lambd=self.lambd)[0][0]
        return loss

class LogisticRegression():
    def __init__(self):
        pass
    # Fit a logistic regression line with data X and y
    # Returns cost history
    def fit(self, X, y, learning_rate=0.001, steps=100, lambd=0.01, epsilon=1e-6, retrain=True, optimization='BFGS'):
        # X is original data, so we should add ones
        # Rows are features, cols are samples
        costs = []
        original = X.copy()
        self.m = X.shape[1]
        X = np.vstack((np.ones((1, self.m)), X))
        self.n = X.shape[0]

        if retrain or not hasattr(self, 'parameters'):
            self.parameters = np.zeros((1, self.n))

        if not hasattr(self, 'optimizer'):
            if optimization == 'gradient_descent':
                self.optimizer = GradientDescentOptimizer(X, y, self.parameters, self.loss, self.gradients, learning_rate=learning_rate, lambd=lambd)
            elif optimization.startswith('newton'):
            # Newton's method, faster, but more computationally expensive
                self.optimizer = NewtonMethodOptimizer(X, y, self.parameters, self.loss, self.gradients, self.second_derivatives, lambd=lambd)
            # BFGS method
            elif optimization == 'BFGS':
                self.optimizer = BFGSOptimizer(X, y, self.parameters, self.loss, self.gradients, lambd=lambd)

        for step in range(steps):
            costs.append(self.optimizer.step())
        
        X = original
        print(self.parameters, costs[-1])
        return costs



    def predict(self, X):
        original = X.copy()
        m = X.shape[1]
        X = np.vstack((np.ones((1, m)), X))
        prediction = sigmoid(np.dot(self.parameters, X)) > 0.5
        X = original
        return prediction

    @staticmethod
    def loss(X, y, parameters, lambd):
        m = X.shape[1]
        g = sigmoid(np.dot(parameters, X))
        # Aviod divided by zeros
        epsilon = 1e-8
        J = -(np.dot(np.log(g + epsilon), y.T) + np.dot(np.log(1 - g + epsilon), (1 - y).T)) / m
        J += lambd / (2 * m) * np.linalg.norm(parameters[0, 1:]) ** 2
        return J

    @staticmethod    
    def gradients(X, y, parameters, lambd):
        m = X.shape[1]
        gradients = np.dot((sigmoid(np.dot(parameters, X)) - y), X.T) / m
        gradients += np.hstack((0, lambd / m * gradients[0, 1:]))
        return gradients

    @staticmethod
    def second_derivatives(X, y, parameters, lambd):
        n, m = X.shape
        h = sigmoid(np.dot(parameters, X))
        H = np.zeros((n, n))
        # Second derivative
        # Didn't figure out the vectorized version, TODO
        for i in range(m):
            sample = X[:, i]
            sample = sample.reshape(n, 1)
            # Need to correct the equation here
            H += (np.dot(sample, sample.T) * h[0][i] * (1 - h[0][i]))
        return H


def sigmoid(X):
    # More stable
    return .5 * (1 + np.tanh(.5 * X))
    # return 1 / (1 + np.exp(-X))

# Line Search
def line_search(X, y, f):
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
    f1 = f(alpha_1)
    f2 = f(alpha_2)
    while True:
        # Step 2
        if f1 > f2:
            h = t * h
        else:
            h = -h
            alpha_1, alpha_2 = alpha_2, alpha_1
            f1, f2 = f2, f1
        alpha_3 = alpha_2 + h
        f3 = f(alpha_3)
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
    f_l = f(lambda_k)
    f_m = f(mu_k)
    while True:
        if f_l > f_m:
            if b_k - lambda_k <= delta:
                return mu_k
            else:
                a_k = lambda_k
                lambda_k = mu_k
                f_l = f_m
                mu_k = a_k + 0.618 * (b_k - a_k)
                f_m = f(mu_k)
        else:
            if mu_k - a_k <= delta:
                return lambda_k
            else:
                b_k = mu_k
                mu_k = lambda_k
                f_m = f_l
                lambda_k = a_k + 0.382 * (b_k - a_k)
                f_l = f(lambda_k)

def decision_boundary(X, y, lr):
    plt.clf()
    delta = 0.25
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
        lr.fit(X, y, learning_rate=0.0000002, steps=1,  retrain=False, optimization = 'BFGS')
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