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

    #As per q1.py, I attempted to use np.genfromtxt('Frogs-subsample.csv',delimiter=',',dtype=('unicode'),skip_header=1)
    #  however, ran into error 'numpy.ndarray' object has no attribute 'to_numpy'. Ergo, why I used pd.
    def __init__(self):
        self.frogsCSV = pd.read_csv('Frogs.csv', header=0, index_col=False)
        self.frogsSampleCSV = pd.read_csv('Frogs-subsample.csv', header=0,
                                          index_col=False)

    #   Since the task here is to create a perceptron using a sigmoid activation function,
    #   I found websites with examples of such and used them as inspiration for the following.
    #   Website : https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110
    def sigmoidActivation(self, x):
        return 1 / (1 + np.exp(-x))

    def logLossFxn(self, y, output):
        return -1 * (y * np.log(output) + (1 - y) * np.log(1 - output))

    def outputFxn(self, features, weights, bias):
        return self.sigmoidActivation(np.dot(features, weights) + bias)

    def updateWeights(self, x, y, weights, bias, learn_rate):
        for index in range(len(weights)):
            weights[index] += learn_rate * (y - self.outputFxn(x, weights, bias)) * x[index]
        bias += learn_rate * (y - self.outputFxn(x, weights, bias))
        return weights, bias


    # Source: For drawing these plots I found https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    # helped quite a bit. I thought making the edgecolor black helps in seeing the individual dots better on the plot.
    def drawScatterPlots(self, X, y):
        admitted = X[np.argwhere(y == 1)]
        rejected = X[np.argwhere(y == 0)]
        plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], color='magenta', edgecolor='black',
                    label='Hyla minuta')
        plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], color='orange', edgecolor='black',
                    label='Hypsiboas cinerascens')
        plt.xlabel('MFCCs_10 Values')
        plt.ylabel('MFCCs_17 Values')

    def drawDecisionBoundary(self, m, b, color='b-.'):

        x = np.arange(-0.5, 0.5, 0.1)
        plt.plot(x, m * x + b, color, label='Decision Boundary')

    def trainDataSet(self, features, targets, epochs, learningRate, graph_lines=False):
        np.random.seed(1)
        numberOfEpochs = 150
        learningRate = 0.01
        errorArray = []
        numberOfRecords, numberOfFeatures = features.shape
        previousLoss = None
        weights = np.random.normal(scale=1 / numberOfFeatures ** 0.5, size=numberOfFeatures)
        bias = 0
        for e in range(numberOfEpochs):
            for x, y in zip(features, targets):
                output = self.outputFxn(x, weights, bias)
                weights, bias = self.updateWeights(x, y, weights, bias, learningRate)

            # Printing out the log-loss error on the training set
            out = self.outputFxn(features, weights, bias)

            loss = np.mean(self.logLossFxn(targets, out))
            errorArray.append(loss)
            if e % (epochs / 10) == 0:
                print("\n--- Epoch", e, "---")
                if previousLoss and previousLoss < loss:
                    print("Training loss: ", loss)
                else:
                    print("Training loss: ", loss)
                previousLoss = loss
                predictions = out > 0.5
                accuracy = np.mean(predictions == targets)
                print("Accuracy: ", accuracy)
            if graph_lines and e % (epochs / 150) == 0:
                self.drawDecisionBoundary(-weights[0] / weights[1], -bias / weights[1])

        # Here, I'm drawing the decision boundary for Frogs.CSV and Frogs_subsample.CSV.
        plt.title("Decision Boundary Graph")
        self.drawDecisionBoundary(-weights[0] / weights[1], -bias / weights[1], 'black')

        # Here, I'm drawing Scatter Plots for Frogs.CSV and Frogs_subsample.CSV.
        self.drawScatterPlots(features, targets)
        plt.show()

        # Here, I'm showing how the error rate changes as the epochs increase.
        plt.title("Error Plot")
        plt.xlabel('Epoch #')
        plt.ylabel('Error')
        plt.plot(errorArray)
        plt.show()

    def executePerceptron(self):
        data = self.frogsSampleCSV.to_numpy()
        np.random.shuffle(data)

        X = data[:, :2]
        X = X.astype(np.float64)
        y = np.unique(data[:, 2], return_inverse=True)[1]
        numberOfEpochs = 150
        learningRate = 0.01
        self.trainDataSet(X, y, numberOfEpochs, learningRate, True)

        data = self.frogsCSV.to_numpy()
        np.random.shuffle(data)

        X = data[:, :2]
        X = X.astype(np.float64)
        y = np.unique(data[:, 2], return_inverse=True)[1]
        numberOfEpochs = 150
        learningRate = 0.01
        self.trainDataSet(X, y, numberOfEpochs, learningRate, True)


#   Creates an object from the BinaryClassifier class so we can perform binary classification
#   and generate the graphs as per question 2's requirements.
if __name__ == '__main__':
    perceptron = BinaryClassifier()
    perceptron.executePerceptron()