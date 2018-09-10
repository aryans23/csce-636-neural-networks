import numpy as np
import sys

"""This script implements a two-class perceptron model.
"""

class perceptron(object):
	
	def __init__(self, max_iter):
		self.max_iter = max_iter

	def fit(self, X, y):
		"""Train perceptron model on data (X,y).

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE

		self.W = np.array([0.0 for i in range(X.shape[1])])
		predicted = np.zeros(y.shape[0])
		self.errors = []
		for it in range(self.max_iter):
			for i in np.arange(X.shape[0]):
				act = np.sum(np.dot(X[i], self.W))
				yhat = 1 if act >= 0 else -1
				predicted[i] = yhat
				self.W = self.W + (y[i]-yhat) * X[i]
			error = np.sum(np.array([(predicted[i]-y[i])**2 for i in range(y.shape[0])]))
			self.errors.append(error)

		### END YOUR CODE
		
		return self

	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W

	def predict(self, X):
		"""Predict class labels for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""
		### YOUR CODE HERE
		
		preds = np.ndarray(X.shape[0])
		for i in np.arange(X.shape[0]):
			act = np.sum(np.dot(X[i], self.W))
			preds[i] = 1 if act >= 0 else -1
		return preds

		### END YOUR CODE

	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			score: An float. Mean accuracy of self.predict(X) wrt. y.
		"""
		### YOUR CODE HERE

		preds = self.predict(X)
		from sklearn.metrics import accuracy_score
		return accuracy_score(preds, y)*100

		### END YOUR CODE


