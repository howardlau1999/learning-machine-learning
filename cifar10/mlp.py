import numpy as np


def softmax(x):
    e_x = np.exp(x - np.reshape(np.max(x, axis=1), (x.shape[0], 1)))
    return e_x / e_x.sum(axis=1, keepdims=True)


def log_softmax(x):
    return np.log(softmax(x))


def ReLU(x):
    return np.maximum(0, x)


def add_ones(x):
    return np.hstack([np.ones((x.shape[0], 1)), x])


def nll_loss(logits, ground_truths):
    return np.mean(-logits[range(len(ground_truths)), ground_truths])


def cross_entropy_loss(outputs, ground_truths):
    return nll_loss(log_softmax(outputs), ground_truths)


def delta_cross_entropy(outputs, ground_truths):
    m = len(ground_truths)
    grad = softmax(outputs)
    grad[range(m), ground_truths] -= 1
    return grad / m


def delta_relu(x):
    delta = np.ones(x.shape)
    delta[x <= 0] = 0
    return delta


class MLP:
    def __init__(self, input_size=3072, hidden_size=1000, output_size=10):
        self.input_hidden = np.random.uniform(-0.001,
                                              0.001, (input_size + 1, hidden_size))
        self.input_hidden[0, :] = 0
        self.hidden_output = np.random.uniform(-0.001,
                                               0.001, (hidden_size + 1, output_size))
        self.hidden_output[0, :] = 0
        self.z = None
        self.a = None
        self.o = None

    def fit(self, batch_size, epochs, lambd, learning_rate, decay, train_images, train_labels, test_images,
            test_labels):
        loss_history = []
        train_acc = []
        test_acc = []
        for epoch in range(epochs):
            train_idx = np.random.choice(
                len(train_images), len(train_images), replace=False)
            batches = int(np.ceil(len(train_labels) / batch_size))
            for batch in range(batches):
                begin = batch * batch_size
                end = (batch + 1) * batch_size
                X = train_images[train_idx[begin:end]]
                y = train_labels[train_idx[begin:end]]
                outputs = self.predict(X)
                loss = cross_entropy_loss(outputs, y) + lambd / 2 * (
                    np.sum(self.input_hidden[1:, :] ** 2) + np.sum(self.hidden_output[1:, :] ** 2))
                loss_history.append(loss)
                predictions = np.argmax(softmax(outputs), axis=1)
                train_accuracy = np.mean(predictions == y)
                train_acc.append(train_accuracy)

                if batch % 50 == 0:
                    print(
                        f"Epoch {epoch + 1} Batch {batch + 1}/{batches} Loss {loss:.4f}")

                # Backward
                delta_output = delta_cross_entropy(
                    outputs, y)  # batch_size x output_size
                delta_hidden = add_ones(
                    self.a).T @ delta_output + lambd * self.hidden_output  # hidden_size + 1 x output_size
                delta_hidden[0, :] -= lambd * self.hidden_output[0, :]
                delta_input = add_ones(X).T @ (delta_relu(
                    self.z) * (delta_output @ self.hidden_output[1:, :].T)) + lambd * self.input_hidden  # input_size + 1 x hidden_size
                delta_input[0, :] -= lambd * self.input_hidden[0, :]

                # Optimize
                self.input_hidden -= learning_rate * delta_input
                self.hidden_output -= learning_rate * delta_hidden

            learning_rate *= decay
            # Evaluation
            predictions = np.argmax(softmax(self.predict(test_images)), axis=1)
            accuracy = np.mean(predictions == test_labels)
            test_acc.append(accuracy)
            print(
                f"Epoch {epoch + 1} Test Accuracy: {accuracy * 100:.2f}% Train Accuracy: {np.mean(train_acc[-batches:]) * 100:.2f}%")

        np.save("weights.npy", {
                "input_hidden": self.input_hidden, "hidden_output": self.hidden_output})
        np.save("history.npy", {
                "test_acc": test_acc, "train_acc": train_acc, "loss_history": loss_history})

    def predict(self, images):
        x = add_ones(images)
        self.z = np.matmul(x, self.input_hidden)
        self.a = ReLU(self.z)
        self.o = np.matmul(add_ones(self.a), self.hidden_output)
        return self.o
