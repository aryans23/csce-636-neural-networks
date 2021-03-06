import tensorflow as tf
from Model import MNIST
from DataReader import load_data, train_valid_split

def configure():
	flags = tf.app.flags

	flags.DEFINE_integer('num_hid_layers', 1, 'the number of hidden layers')
	flags.DEFINE_integer('num_hid_units', 512, 'the number of hidden units in hidden layers')
	flags.DEFINE_integer('batch_size', 32, 'training batch size')
	flags.DEFINE_integer('num_classes', 10, 'number of classes')
	flags.DEFINE_string('modeldir', 'model', 'model directory')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	sess = tf.Session()
	print('---Prepare data...')
	x_train, y_train, x_test, y_test = load_data()
	x_train_new, y_train_new, x_valid, y_valid \
				= train_valid_split(x_train, y_train)

	model = MNIST(sess, configure())

	### YOUR CODE HERE
	# First run: use the train_new set and the valid set to choose
	# hyperparameters, like num_hid_layers, num_hid_units, stopping epoch, etc.
	# Report chosen hyperparameters in your hard-copy report.

	for epoch in [1, 5, 10]:
		model.train(x_train_new, y_train_new, x_valid, y_valid, epoch)

	# Second run: with hyperparameters determined in the first run, re-train
	# your model on the original train set.

	model.train(x_train_new, y_train_new, None, None, 9, False)

	# Third run: after re-training, test your model on the test set.
	# Report testing accuracy in your hard-copy report.

	model.test(x_test, y_test, 9)
	
	### END CODE HERE

if __name__ == '__main__':
	tf.app.run()
