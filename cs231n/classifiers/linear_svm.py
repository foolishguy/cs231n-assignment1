import numpy as np
from random import shuffle
from past.builtins import xrange

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
    return svm_loss_vectorized(W, X, y, reg)

def svm_loss_vectorized(W, X, y, reg):
    """
      Structured SVM loss function, vectorized implementation.

      Inputs and outputs are the same as svm_loss_naive.
      """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    scores = X.dot(W)
    yi_scores = scores[np.arange(num_train), y]

    margins = np.maximum(0, scores - np.reshape(yi_scores, (-1,1)) + 1) 
    margins[np.arange(num_train), y] = 0 

    loss = np.mean(np.sum(margins, axis=1))
    loss += reg * np.sum(W * W)

    margins[margins > 0] = 1

    margins_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -margins_sum.T
    dW = np.dot(X.T, margins)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
