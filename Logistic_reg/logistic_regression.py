import numpy as np
from tqdm import tqdm
from sklearn.datasets import make_blobs

class LogisticRegression():
    def __init__(self, train, lr=0.1, num_iters=10000):
        self.lr = lr
        self.num_iters = num_iters

        self.num_trains, self.dims = train.shape

    def train(self, X, y):
        self.weights = np.zeros((self.dims, 1))
        self.bias = 0
        pbar = tqdm(range(self.num_iters))
        for i in pbar:
            y_preds = self.sigmoid(np.dot(X, self.weights) + self.bias)

            loss = -1/self.num_trains * np.sum(y*np.log(y_preds) + (1-y)*np.log(1-y_preds))

            dw = 1/self.num_trains * np.dot(X.T, (y_preds - y))
            db = 1/self.dims * np.sum(y_preds - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 1000 == 0:
                pbar.set_description(f'Loss after iterarion {i}th: {loss}')

        return self.weights, self.bias

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_pred_labels = y_pred > 0.5

        return y_pred_labels

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    np.random.seed(1)

    X, y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis]

    model = LogisticRegression(X)
    w, b = model.train(X, y)
    y_preds = model.predict(X)

    print(f'Accuracy: {np.sum(y == y_preds) / X.shape[0]}')