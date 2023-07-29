import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import load_iris

# Load the data
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
Y = iris["target"]

# add the bias term
X = np.c_[np.ones([len(X), 1]), X]

# split the data manually using the same conecpt of sklearn train_test_split
# we will use 80% of the data for training and 20% for testing
# we will use the same random state to get the same result
np.random.seed(42)
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X)
test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

# shuffle the data
rnd_idx = np.random.permutation(total_size)
X_train = X[rnd_idx[:train_size]]
Y_train = Y[rnd_idx[:train_size]]
X_valid = X[rnd_idx[train_size:-test_size]]
Y_valid = Y[rnd_idx[train_size:-test_size]]
X_test = X[rnd_idx[-test_size:]]
Y_test = Y[rnd_idx[-test_size:]]


# convert the target to a one hot vector using the same concept of sklearn OneHotEncoder
Y_train_one_hot = np.zeros((len(Y_train), max(Y_train) + 1))
Y_train_one_hot[np.arange(len(Y_train)), Y_train] = 1
Y_test_one_hot = np.zeros((len(Y_test), max(Y_test) + 1))
Y_test_one_hot[np.arange(len(Y_test)), Y_test] = 1
Y_valid_one_hot = np.zeros((len(Y_valid), max(Y_valid) + 1))
Y_valid_one_hot[np.arange(len(Y_valid)), Y_valid] = 1


# Scale the Data using the mean and std of the training data
# get the mean and std of the training data without the bias term
mean = X_train[:, 1:].mean(axis=0, keepdims=True)
std = X_train[:, 1:].std(axis=0, keepdims=True)

# scale the training data

X_train[:, 1:] = (X_train[:, 1:] - mean) / std
X_valid[:, 1:] = (X_valid[:, 1:] - mean) / std
X_test[:, 1:] = (X_test[:, 1:] - mean) / std

# write a func simialr to the softmax func in sklearn
def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums

# train the model using the cost function and the gradient descent

eta = 0.01 # learning rate
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

# initialize the theta
theta = np.random.randn(X_train.shape[1], len(np.unique(Y_train)))

for epcho in range(n_iterations):
    logits = X_train @ theta
    Y_proba = softmax(logits)
    loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1)) # cross entropy loss
    error = Y_proba - Y_train_one_hot
    if epcho % 500 == 0:
        print(f"Epcho: {epcho}, Loss: {loss}")
    gradients = 1/m * X_train.T @ error
    theta = theta - eta * gradients

# get the accuracy of the model
logits = X_valid @ theta
Y_proba = softmax(logits)
Y_predict = np.argmax(Y_proba, axis=1)

accuracy = np.mean(Y_predict == Y_valid)
print(f"Accuracy: {accuracy}")


# try using the l2 regularization
theta = np.random.randn(X_train.shape[1], len(np.unique(Y_train)))
alpha = 0.1 # regularization hyperparameter used to control the regularization strength (the higher the stronger) the penalty term weight in the cost function
for epcho in range(n_iterations):
    logits = X_train @ theta
    Y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    error = Y_proba - Y_train_one_hot
    if epcho % 500 == 0:
        print(f"Epcho: {epcho}, Loss: {loss}")
    gradients = 1/m * X_train.T @ error + np.r_[np.zeros([1, len(np.unique(Y_train))]), alpha * theta[1:]]
    theta = theta - eta * gradients

# get the accuracy of the model
logits = X_valid @ theta
Y_proba = softmax(logits)
Y_predict = np.argmax(Y_proba, axis=1)

accuracy = np.mean(Y_predict == Y_valid)
print(f"Accuracy: {accuracy}")


# model with early stopping
eta = 0.01
n_epochs = 50_001
m = len(X_train)
epsilon = 1e-7
C = 100  # regularization hyperparameter

best_loss = np.Infinity
np.random.seed(42)
Theta = np.random.randn(X_train.shape[1], len(np.unique(Y_train)))

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    Y_proba_valid = softmax(X_valid @ Theta)
    xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()
    total_loss = xentropy_losses.sum(axis=1).mean() + 1 / C * l2_loss
    if epoch % 1000 == 0:
        print(epoch, total_loss.round(4))
    if total_loss < best_loss:
        best_loss = total_loss
    else:
        print(epoch - 1, best_loss.round(4))
        print(epoch, total_loss.round(4), "early stopping!")
        break
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, len(np.unique(Y_train))]), 1 / C * Theta[1:]]
    Theta = Theta - eta * gradients
# get the accuracy of the model
logits = X_valid @ theta
Y_proba = softmax(logits)
Y_predict = np.argmax(Y_proba, axis=1)

accuracy = np.mean(Y_predict == Y_valid)
print(f"Accuracy: {accuracy}")



# measure the performance against the test set
logits = X_test @ theta
Y_proba = softmax(logits)
Y_predict = np.argmax(Y_proba, axis=1)

accuracy = np.mean(Y_predict == Y_test)
print(f"Accuracy: {accuracy}")

