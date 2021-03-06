import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def unpickle(file_path):
	with open(file_path, mode='rb') as file:
		data = pickle.load(file, encoding='bytes')
	return data

def load_data(data_dir):
	"""Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""

	### YOUR CODE HERE

	x_train_list = []
	y_train_list = []
	x_test_list = []
	y_test_list = []

	for filename in os.listdir(data_dir):
		if filename.startswith("data"):
			data = unpickle(data_dir+"/"+filename)
			raw_images = list(data[b'data'])
			cls = data[b'labels']
			x_train_list = x_train_list + raw_images
			y_train_list = y_train_list + cls
		if filename.startswith("test"):
			data = unpickle(data_dir+"/"+filename)
			raw_images = list(data[b'data'])
			cls = data[b'labels']
			x_test_list = x_test_list + raw_images
			y_test_list = y_test_list + cls
	
	x_train = np.array(x_train_list)
	y_train = np.array(y_train_list)
	x_test = np.array(x_test_list)
	y_test = np.array(y_test_list)

	### END CODE HERE

	return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
	"""Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid

