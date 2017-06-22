import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  C = W.shape[1]
    
  for i in range(num_train):
    feature = X[i,:]
    scores = feature.dot(W)
    #numeric_stability_factor = np.max(scores)
    scores -= np.max(scores)
    #scores_exp = np.exp(scores)
    #loss += -np.log(scores_exp[y[i]]) + np.log(np.sum(scores_exp))
    scores_prob = np.exp(scores) / np.sum(np.exp(scores))
    loss += -np.log(scores_prob[y[i]])
    
    for j in range(C):
        if j==y[i]:
            dW[:,j] += (scores_prob[j] - 1)*feature
        else:
            dW[:,j] += scores_prob[j]*feature
  
  loss /= num_train
  loss += reg*np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  scores = X.dot(W)
  scores -= np.max(scores)
  scores = (np.exp(scores.T) / np.sum(np.exp(scores).T, axis=0)).T
  loss_prob = -np.log(scores[np.arange(X.shape[0]),y])
  loss = np.sum(loss_prob)/X.shape[0]
  loss += reg*np.sum(W*W) 
  
  scores[np.arange(X.shape[0]),y] = (scores[np.arange(X.shape[0]),y] - 1)
  dW = (scores.T.dot(X)).T
  dW /= X.shape[0]
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

