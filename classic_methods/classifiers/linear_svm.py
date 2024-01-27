from builtins import range
import numpy as np
from random import shuffle
import cupy as cp

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = cp.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = cp.dot(X[i], W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i] # gradient update for incorrect rows
                dW[:,y[i]] -= X[i] # gradient update for correct rows

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW = dW / num_train
    # Add regularization to the loss.
    loss += reg * cp.sum(W * W)
    dW += 2 * reg * W # regularization gradient update

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = cp.zeros(W.shape) # initialize the gradient as zero

    scores = cp.dot(X, W) # N x C
    correct_class_scores = scores[cp.arange(X.shape[0]), y] # N x 1
    margins = cp.maximum(0, scores - correct_class_scores[:, cp.newaxis] + 1) # N x C
    margins[cp.arange(X.shape[0]), y] = 0 # set correct class margins to 0
    loss = cp.sum(margins) / X.shape[0] # average loss
    loss += reg * cp.sum(W * W) # regularization

    binary = cp.zeros(margins.shape) # N x C
    binary[margins > 0] = 1 # N x C (binary mask)
    row_sum = cp.sum(binary, axis=1) # N x 1
    binary[cp.arange(X.shape[0]), y] = -row_sum # N x C
    dW = cp.dot(X.T, binary) / X.shape[0] # D x C
    dW += 2 * reg * W # regularization gradient update

    return loss, dW
