# Author: Michael Walia - mpw2217
# Class: 635 - Introduction to Machine Learning
# Date: 11-16-2021
# Homework 3: Multiclass Classification
# Problem: q1a.py

import pandas as pd
import numpy as np
import os

# This numerical gradient algorithm calculates an approximation of nabla.
# It follows the 8 steps listed on page 3 of the homework.

# The following notation is used in this assignment (the below quote is directly from the homework)
# "Our parameters are housed in the construct Θ = {W, b} (where W is a
# weight matrix of dimensions (d × k) and b is a bias vector of dimensions (1 × k), where d is the
# number of observed variables and k is the number of classes we want to predict). Our dataset is X
# which is of dimension (N × d) (N is the number of samples) and y (which is a column vector of noting that we have written a regularized loss (imposing a Gaussian prior distribution over the
# input-to-hidden parameters W). The computation of p is simply the application of the softmax link
# function presented to you in class, where f = W · X + b is the computation of the logits (or log
# odds/class scores)."
def numericalGradientAlgorithm(dataSetX, columnVectorY, theta, regularizationNum):
    epsilonNum = 0.0001 # The HW suggests a value of 0.001 or 0.0001. I found no difference using either.
    nablaArray = []
    for index in range(len(theta)):
        parameter = theta[index]
        gradient = np.zeros(parameter.shape)
        iterator = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not iterator.finished:
            # parameters at (x+epsilon) and (x-epsilon)
            thetaPlusEpsilon = theta
            thetaMinusEpsilon = theta
            indexMulti = iterator.multi_index
            # 1. Add ϵ to Θj, put Θj back into Θ, giving us (Θ + ϵ) at j
            thetaPlusEpsilon[index][indexMulti] = parameter[indexMulti] + epsilonNum
            # 2. Given perturbed Θj (all the rest of the values in Θ are fixed), calculate J (Θ + ϵ)
            perturbedThetaPlusEpsilon = calculateCost(dataSetX, columnVectorY, thetaPlusEpsilon, regularizationNum)
            # 3. Resetting theta to its original state.
            theta[index][indexMulti] = parameter[indexMulti] - epsilonNum
            # 4. Subtract ϵ from Θj
            thetaMinusEpsilon[index][indexMulti] = parameter[indexMulti] - epsilonNum
            # 5. Given perturbed Θj, calculate J (Θ − ϵ)
            perturbedThetaMinusEpsilon = calculateCost(dataSetX, columnVectorY, thetaMinusEpsilon, regularizationNum)
            # 6. Estimate the scalar derivative using Equation 5
            # 7. Reset b to its original state, returning ∼ ∇Θ
            theta[index][indexMulti] = parameter[indexMulti] + epsilonNum
            gradient[indexMulti] = (perturbedThetaPlusEpsilon - perturbedThetaMinusEpsilon) / (2 * epsilonNum)
            # 8. Repeat Steps 1-7 for each Θj in Θ
            iterator.iternext()
        nablaArray.append(gradient)
    return tuple(nablaArray)

# This function calculates the softmax loss.
# I wasn't sure how to calculate softmax from scratch so I used this site as a
# reference: https://www.cristiandima.com/neural-networks-from-scratch-in-python/
def calculateLossSoftMax(dataSetX, columnVectorY):
    # This is the forward pass.
    numbersShaped = dataSetX.shape[0]
    exponentionalValues = np.exp(dataSetX)
    probability = exponentionalValues / np.sum(exponentionalValues, axis=1, keepdims=True)
    lossCalculation = -np.mean(np.log(probability[range(numbersShaped), columnVectorY]))
    # This is the backward pass.
    derivativeX = np.array(probability, copy=True)
    derivativeX[range(numbersShaped), columnVectorY] -= 1
    derivativeX /= numbersShaped
    return lossCalculation, probability, derivativeX

# This function computes the gradient descent.
def calculateGradientDescent(dataSetX, columnVectorY, theta, regularizationNum):
    weightMatrix, biasVector = theta[0], theta[1]
    f = dataSetX.dot(weightMatrix) + biasVector
    _, _, df = calculateLossSoftMax(f, columnVectorY)
    derivativeWeightMatrix = np.dot(dataSetX.T, df) + regularizationNum * weightMatrix
    derivativeBiasVector = np.sum(df, axis=0)
    return (derivativeWeightMatrix, derivativeBiasVector)

# This function is the cost function.
def calculateCost(dataSetX, columnVectorY, theta, regularizationNum):
    weightMatrix, biasVector = theta[0], theta[1]
    f = dataSetX.dot(weightMatrix) + biasVector
    dataLoss, _, _ = calculateLossSoftMax(f, columnVectorY)
    regularizationLoss = 0.5 * regularizationNum * np.sum(weightMatrix ** 2)
    calculatedCost = dataLoss + regularizationLoss
    return calculatedCost

# This returns the classification for XOR - 1 or 0.
def classifyXOR(dataSetX, theta):
    weightMatrix, biasVector = theta[0], theta[1]
    XORclass = dataSetX.dot(weightMatrix) + biasVector
    _, probability, _ = calculateLossSoftMax(XORclass, y)
    return XORclass, probability



# Note this requires your XOR.dat to be in the same directory as the q1.py or this won't run.
# I've also included this in my Readme.txt file.
fileLocation = os.getcwd() + '\\xor.dat'
file = pd.read_csv(fileLocation, header=None)

# set X (training data) and y (target variable)
column = file.shape[1]
X = file.iloc[:, 0:column - 1]
y = file.iloc[:, column - 1:column]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)
y = y.flatten()

# Train a Linear Classifier

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters in such a way to play nicely with the gradient-check!
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K)) + 1.0
theta = (W, b)

# Hyperparameters used
regularization = 1e-3  # regularization strength
nablaList = numericalGradientAlgorithm(X, y, theta, regularization)
nabla = calculateGradientDescent(X, y, theta, regularization)
nablaList = list(nablaList)
nabla = list(nabla)

for index in range(0, len(nabla)):
    is_incorrect = 0  # set to false
    gradient = nabla[index]
    gradientList = nablaList[index]
    error = np.linalg.norm(gradientList - gradient) / (np.linalg.norm(gradientList + gradient))
    if (error > 1e-8):
        print("Param {0} is WRONG, error = {1}".format(index, error))
    else:
        print("Param {0} is CORRECT, error = {1}".format(index, error))

# Re-initialize parameters for generic training
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))
theta = (W, b)

numberOfEpochs = 1000
check = 100  # every so many pass/epochs, print loss/error to terminal
stepSize = 0.1
regularization = 0.1  # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(numberOfEpochs):
    # Here I perform one step of gradient descent.
    theta = (W, b)
    loss = calculateCost(X, y, theta, regularization)
    if i % check == 0:
        print("Iteration [%d] - Loss [%f]" % (i, loss))

    # Here I update the parameter.
    dW, db = calculateGradientDescent(X, y, theta, regularization)
    W = W - stepSize * dW
    b = b - stepSize * db


# Here I see what the training set accuracy is.
scores, probability = classifyXOR(X, theta)
classPrediction = np.argmax(scores, axis=1)
print('Training Accuracy: %.2f%%' % (100 * np.mean(classPrediction == y)))