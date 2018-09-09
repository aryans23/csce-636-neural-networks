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
	print("raw_image shape = " + str(raw_image.shape))

	# Feature 1: Measure of Symmetry
	### YOUR CODE HERE

	flip_X = np.fliplr(raw_image)
	diff = np.abs(raw_image - flip_X)
	feature_1 = -np.sum(np.sum(diff, axis=1), axis=1)/256
	print("feature_1 shape= " + str(feature_1.shape))

	### END YOUR CODE

	# Feature 2: Measure of Intensity
	### YOUR CODE HERE

	feature_2 = np.sum(np.sum(raw_image, axis=1), axis=1)/256
	print("feature_2 shape= " + str(feature_2.shape))
	
	### END YOUR CODE

	# Feature 3: Bias Term. Always 1.
	### YOUR CODE HERE

	feature_3 = np.ones(raw_X.shape[0])
	print("feature_3 shape= " + str(feature_3.shape))
	
	### END YOUR CODE

	# Stack features together in the following order.
	# [Feature 3, Feature 1, Feature 2]
	### YOUR CODE HERE
	X = np.stack([feature_3, feature_2, feature_1], axis=1)
	print("** X: ")
	print(X.view())
	print("** X ends")
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

	y = np.zeros(raw_y.shape[0])

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

	print("raw_X shape = " + str(raw_X.shape))
	print("raw_y shape = " + str(raw_y.shape))

	assert len(raw_X.shape) == 2
	assert raw_X.shape[0] == raw_y.shape[0]

	return prepare_X(raw_X), prepare_y(raw_y)



