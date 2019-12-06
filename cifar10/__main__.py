from .knn import KNN
from .dataloader import DataLoader
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from .mlp import MLP
from .svm import SVM

loader = DataLoader()
loader.load_data("cifar10/cifar-10-python/cifar-10-batches-py")

def eval_svm():
    svm = SVM()
    mean_image = np.mean(loader.train_images, axis=0)
    svm.fit(learning_rate=1e-3, delta=1, reg=0.5, train_labels=loader.train_labels, train_images=loader.train_images - mean_image, batch_size=200, epochs=300)
    predictions = np.argmax(svm.predict(loader.test_images - mean_image), axis=1)
    accuracy = np.mean(predictions == loader.test_labels)
    print(f"SVM Test Accuracy: {accuracy * 100:.2f}%")

# eval_svm()

def eval_mlp():
    mlp = MLP(hidden_size=100)
    mean_image = np.mean(loader.train_images, axis=0)
    mlp.fit(batch_size=200, epochs=300, learning_rate=1e-4, decay=0.95, lambd=0.5, train_images=loader.train_images - mean_image,
            train_labels=loader.train_labels, test_images=loader.test_images - mean_image,
            test_labels=loader.test_labels)


eval_mlp()


def eval_knn():
    
    train_num = 10000
    test_num = 500
    for k in [3, 5, 7, 9, 11, 21, 51, 101]:
        knn = KNN(k)
        pool = Pool(4)

        train_idx = np.random.choice(len(loader.train_images), train_num, replace=False)
        test_idx = np.random.choice(len(loader.test_images), test_num, replace=False)

        knn.fit(loader.train_images[train_idx], loader.train_labels[train_idx])
        predictions = pool.map(knn.predict, loader.test_images[test_idx])
        pool.close()
        pool.join()
        knn_accuracy = np.sum(predictions == loader.test_labels[test_idx]) / test_num
        print(f"KNN(k = {k}) Accuracy: {knn_accuracy * 100:.2f}%")

# eval_knn()