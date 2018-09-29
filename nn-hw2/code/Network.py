import tensorflow as tf

"""This script defines the network.
"""

class MLP(object):

	def __init__(self, num_hid_layers, num_hid_units, num_classes):
		"""Define hyperparameters.

		Args:
			num_hid_layers: A positive integer.
				Define the number of hidden layers.
			num_hid_units: A positive integers. 
				Define the number of hidden units in hidden layers.
			num_classes: A positive integer.
		"""
		self.num_hid_layers = num_hid_layers
		self.num_hid_units = num_hid_units
		self.num_classes = num_classes

	def __call__(self, inputs, training):
		"""Add operations to classify a batch of input images.

		Args:
			inputs: A Tensor representing a batch of input images.
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			A logits Tensor with shape [<batch_size>, self.num_classes].
		"""
		outputs = self._hidden_layers(inputs, training)

		return self._output_layer(outputs, training)

	################################################################################
	# Blocks building the network
	################################################################################
	def _hidden_layers(self, inputs, training):
		"""Implement the hidden layers according to self.num_hid_layers
		and self.num_hid_units.

		Args:
			inputs: A Tensor with shape [<batch_size>, 784].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A Tensor with shape [<batch_size>, self.num_hid_units].
		"""
		
		### YOUR CODE HERE
		# Note: for tensorflow APIs, only those in tf.layers and tf.nn
		# are allowed to use.

		W1 = tf.Variable(tf.random_normal([784, self.num_hid_units], stddev=0.03, dtype=tf.float64), name='W1')
		b1 = tf.Variable(tf.random_normal([self.num_hid_units], dtype=tf.float64), name='b1')
		h1 = tf.add(tf.matmul(inputs, W1), b1)
		outputs = tf.nn.relu(h1)

		### END CODE HERE

		return outputs

	def _output_layer(self, inputs, training):
		"""Implement the output layer.

		Args:
			inputs: A Tensor with shape [<batch_size>, self.num_hid_units].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A logits Tensor with shape [<batch_size>, self.num_classes].
		"""

		### YOUR CODE HERE
		# Note: for tensorflow APIs, only those in tf.layers and tf.nn
		# are allowed to use.

		W2 = tf.Variable(tf.random_normal([self.num_hid_units, self.num_classes], stddev=0.03, dtype=tf.float64), name='W2')
		b2 = tf.Variable(tf.random_normal([self.num_classes], dtype=tf.float64), name='b2')
		outputs = tf.nn.softmax(tf.add(tf.matmul(inputs, W2), b2))
		
		### END CODE HERE

		return outputs

















