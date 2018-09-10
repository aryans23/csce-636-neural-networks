import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
	def __init__(self, learning_rate, max_iter):
		self.learning_rate = learning_rate
		self.max_iter = max_iter

	def _sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def fit_GD(self, X, y):
		"""Train perceptron model on data (X,y) with GD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE

		self.W = np.zeros(X.shape[1])
		self.errors = np.ndarray(self.max_iter)
		for it in range(self.max_iter):
			grads = np.zeros(X.shape[1])
			for i in range(X.shape[0]):
				grads += self._gradient(X[i], y[i])
			self.W = self.W + self.learning_rate * grads
			self.errors[it] = 100 - self.score(X,y)

		### END YOUR CODE

		return self

	def fit_BGD(self, X, y, batch_size):
		"""Train perceptron model on data (X,y) with BGD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.
			batch_size: An integer.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE

		self.W = np.zeros(X.shape[1])
		self.errors = np.ndarray(self.max_iter)
		for it in range(self.max_iter):
			grads = np.zeros(X.shape[1])
			shuf = np.c_[X,y]
			np.random.shuffle(shuf)
			X = shuf[:,:-1]
			y = shuf[:,-1]
			for i in range(batch_size):
				grads += self._gradient(X[i], y[i])
			self.W = self.W + self.learning_rate * grads
			self.errors[it] = 100 - self.score(X,y)

		### END YOUR CODE

		return self

	def fit_SGD(self, X, y):
		"""Train perceptron model on data (X,y) with SGD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE

		self.W = np.zeros(X.shape[1])
		self.errors = np.ndarray(self.max_iter)
		for it in range(self.max_iter):
			grads = np.zeros(X.shape[1])
			for i in range(X.shape[0]):
				grads += self._gradient(X[i], y[i])
				self.W = self.W + self.learning_rate * grads
			self.errors[it] = 100 - self.score(X,y)

		### END YOUR CODE
		
		return self

	def _gradient(self, _x, _y):
		"""Compute the gradient of cross-entropy with respect to self.W
		for one training sample (_x, _y). This function is used in SGD.

		Args:
			_x: An array of shape [n_features,].
			_y: An integer. 1 or -1.

		Returns:
			_g: An array of shape [n_features,]. The gradient of
				cross-entropy with respect to self.W.
		"""
		### YOUR CODE HERE
		
		z = np.dot(self.W.T, _x)
		return (_y * _x)/(1 + np.exp(_y * z))

		### END YOUR CODE

	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W

	def predict_proba(self, X):
		"""Predict class probabilities for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds_proba: An array of shape [n_samples, 2].
				Only contains floats between [0,1].
		"""
		### YOUR CODE HERE

		yhat = np.dot(X,self.W)
		proba = np.c_[yhat, 1-yhat]
		return proba

		### END YOUR CODE

	def predict(self, X):
		"""Predict class labels for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""
		### YOUR CODE HERE

		yhat = np.dot(X,self.W)
		preds = [1 if x >= 0.5 else -1 for x in self._sigmoid(yhat)]
		return np.array(preds)

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


