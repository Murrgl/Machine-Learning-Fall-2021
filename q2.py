# Author - Michael Walia - mpw2217
# Class - 635 Introduction to Machine Learning @ Rochester Institute of Technology
# Date Created - 10-1-2021 1126

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Question 2 - The Effect of Training Data

# First, creating a perceptron which uses the sigmoid activation function.
# Second, creating scatter plots for Frogs.csv and Frogs_subsample.csv.
# Third, visualizing class regions and decision boundaries.

class BinaryClassifier(object):

    #As per q1, I attempted to use np.genfromtxt('Frogs-subsample.csv',delimiter=',',dtype=('unicode'),skip_header=1)
    #  however, ran into error 'numpy.ndarray' object has no attribute 'to_numpy'. Ergo, why I used pd.
    def __init__(self):
        self.frogsCSV = pd.read_csv('Frogs.csv', header=0, index_col=False)
        self.frogsSampleCSV = pd.read_csv('Frogs-subsample.csv', header=0,
                                          index_col=False)

    #   Since the task here is to create a perceptron using a sigmoid activation function,
    #   I found websites with examples of such and used them as inspiration for the following.
    #   Website : https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def log_loss(self, y, output):
        return -1 * (y * np.log(output) + (1 - y) * np.log(1 - output))

    def output_formula(self, features, weights, bias):
        return self.sigmoid(np.dot(features, weights) + bias)

    def update_weights(self, x, y, weights, bias, learn_rate):
        for index in range(len(weights)):
            weights[index] += learn_rate * (y - self.output_formula(x, weights, bias)) * x[index]
        bias += learn_rate * (y - self.output_formula(x, weights, bias))
        return weights, bias

    def plot_points(self, X, y):
        admitted = X[np.argwhere(y == 1)]
        rejected = X[np.argwhere(y == 0)]
        plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], color='blue', edgecolor='k',
                    label='Hyla minuta')
        plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], color='red', edgecolor='k',
                    label='Hypsiboas cinerascens')

    def display(self, m, b, color='g--'):
        #         plt.xlim(-0.05,1.05)
        #         plt.ylim(-0.05,1.05)
        x = np.arange(-0.5, 0.5, 0.1)
        plt.plot(x, m * x + b, color, label='Decision Boundary')

    def train(self, features, targets, epochs, learn_rate, graph_lines=False):
        np.random.seed(1)
        n_epochs = 100
        learn_rate = 0.01
        errors = []
        n_records, n_features = features.shape
        last_loss = None
        weights = np.random.normal(scale=1 / n_features ** 0.5, size=n_features)
        bias = 0
        for e in range(n_epochs):
            del_w = np.zeros(weights.shape)
            for x, y in zip(features, targets):
                output = self.output_formula(x, weights, bias)
                error = self.log_loss(y, output)
                weights, bias = self.update_weights(x, y, weights, bias, learn_rate)

            # Printing out the log-loss error on the training set
            out = self.output_formula(features, weights, bias)

            loss = np.mean(self.log_loss(targets, out))
            errors.append(loss)
            if e % (epochs / 10) == 0:
                print("\n========== Epoch", e, "==========")
                if last_loss and last_loss < loss:
                    print("Train loss: ", loss, "  WARNING - Loss Increasing")
                else:
                    print("Train loss: ", loss)
                last_loss = loss
                predictions = out > 0.5
                accuracy = np.mean(predictions == targets)
                print("Accuracy: ", accuracy)
            if graph_lines and e % (epochs / 100) == 0:
                self.display(-weights[0] / weights[1], -bias / weights[1])

        # Plotting the solution boundary
        plt.title("Decision boundary")
        self.display(-weights[0] / weights[1], -bias / weights[1], 'black')

        # Plotting the data
        self.plot_points(features, targets)
        #         self.display(-weights[0]/weights[1], -bias/weights[1], 'black')
        plt.show()

        # Plotting the error
        plt.title("Error Plot")
        plt.xlabel('Number of epochs')
        plt.ylabel('Error')
        plt.plot(errors)
        plt.show()

    def run(self):
        data = self.frogsSampleCSV.to_numpy()
        np.random.shuffle(data)

        X = data[:, :2]
        X = X.astype(np.float64)
        y = np.unique(data[:, 2], return_inverse=True)[1]
        n_epochs = 100
        learnrate = 0.01
        self.train(X, y, n_epochs, learnrate, True)

        data = self.frogsCSV.to_numpy()
        np.random.shuffle(data)

        X = data[:, :2]
        X = X.astype(np.float64)
        y = np.unique(data[:, 2], return_inverse=True)[1]
        n_epochs = 100
        learnrate = 0.01
        self.train(X, y, n_epochs, learnrate, True)


#   Creates an object from the BinaryClassifier class so we can perform binary classification
#   and generate the graphs as per question 2's requirements.
if __name__ == '__main__':
    perceptron = BinaryClassifier()
    perceptron.run()