import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
	"""Load a given txt file.

	Args:
		filename: A string.

	Returns:
		raw_data: An array of shape [n_samples, 257].
	"""
	return np.loadtxt(filename)

def train_valid_split(raw_data, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 257].
		split_index: An integer.

	Returns:
		raw_train: An array of shape [n_train_samples, 257].
		raw_valid: An array of shape [n_valid_samples, 257].
	"""
	return raw_data[:split_index], raw_data[split_index:]

def prepare_X(raw_X):
	"""Extract features from raw_X as required.

	Args:
		raw_X: An array of shape [n_samples, 256].

	Returns:
		X: An array of shape [n_samples, n_features].
	"""
	raw_image = raw_X.reshape((-1, 16, 16))

	# Feature 1: Measure of Symmetry
	### YOUR CODE HERE

	### END YOUR CODE

	# Feature 2: Measure of Intensity
	### YOUR CODE HERE
	
	### END YOUR CODE

	# Feature 3: Bias Term. Always 1.
	### YOUR CODE HERE
	
	### END YOUR CODE

	# Stack features together in the following order.
	# [Feature 3, Feature 1, Feature 2]
	### YOUR CODE HERE
	
	### END YOUR CODE

	return X

def prepare_y(raw_y):
	"""Convert labels to binary labels (1 or -1).

	In this assignment:
		1 -> 1
		5 -> -1

	Args:
		raw_y: An array of shape [n_samples,].

	Returns:
		y: An array of shape [n_samples,].
	"""
	### YOUR CODE HERE

	### END YOUR CODE

	return y

def prepare_data(raw_data):
	"""Prepare data for training or testing.

	Args:
		raw_data: An array of shape [n_samples, 257].

	Returns:
		X: An array of shape [n_samples, n_features].
		y: An array of shape [n_samples,].
	"""
	raw_X = raw_data[:, 1:]
	raw_y = raw_data[:, 0]

	assert len(raw_X.shape) == 2
	assert raw_X.shape[0] == raw_y.shape[0]

	return prepare_X(raw_X), prepare_y(raw_y)



