from data_loader import DataLoader
from naive_bayes import NaiveBayes
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Naive Bayes for Spam Classification')

    parser.add_argument(
        '--suffix', default="", type=str,
        help='Dataset to be used for training')

    parser.add_argument(
        '--alpha', default=0.001, type=float,
        help='Smoothing factor of the model')

    args = parser.parse_args()

    # Load data
    data = DataLoader()
    data.load_data(args.suffix)

    # Initialize model
    model = NaiveBayes(data.vocab_size, args.alpha)

    # Train
    model.fit(data.trainX, data.trainY)

    # Evaluation
    predictions = []
    tp, tn, fp, fn = 0, 0, 0, 0
    for (x, y) in zip(data.testX, data.testY):
        probs = model.predict(x)
        labels = list(probs.keys())
        probs = list(probs.values())
        label = labels[np.argmax(probs)]
        if label == 0 and y == 0:
            tn += 1
        elif label == 1 and y == 1:
            tp += 1
        elif label == 0 and y == 1:
            fn += 1
        elif label == 1 and y == 0:
            fp += 1
        predictions.append(label)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Accuracy: {(tp + tn) / len(data.testY) * 100:.2f}%")
    print(f"F1 Score: {2 * (precision * recall) / (precision + recall)}")


if __name__ == '__main__':
    main()
