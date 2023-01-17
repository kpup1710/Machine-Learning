import numpy as np

class KNearestNeighbor():
    def __init__(self, k):
        self.k = k

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        distances = self.comput_distance(x_test)
        return self.predict_labels(distances)
    

    def comput_distance(self, x_test):
        distances = np.sum(x_test**2, axis=1) - 2*np.dot(x_test,self.x_train.T) + np.sum(self.x_train**2,axis=1, keepdims=True)
        return distances

    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i,:])
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))
        return y_pred

if __name__ == '__main__':
    X = np.loadtxt('.\\Test_data\\data.txt', delimiter=',')
    y = np.loadtxt('./Test_data\\targets.txt')

    KNN = KNearestNeighbor(k=3)
    KNN.train(X, y)  
    y_pred = KNN.predict(X)

    print(f'Accuracy: {sum(y_pred == y)/y.shape[0]}')