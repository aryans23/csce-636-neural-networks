import os
import matplotlib.pyplot as plt
from Perceptron import perceptron
from LogisticRegression import logistic_regression
from DataReader import *

data_dir = "../data/"
train_filename = "train.txt"
test_filename = "test.txt"
img_index = 0

def visualize_features(X, y):
	'''This function is used to plot a 2-D scatter plot of training features. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
	
	Returns:
		No return. Save the plot to 'train_features.*' and include it
		in submission.
	'''
	### YOUR CODE HERE

	global img_index
	ones = []
	fives = []
	for i in range(y.shape[0]):
		if y[i]==1:
			ones.append(X[i])
		else:
			fives.append(X[i])
	ones = np.array(ones)
	fives = np.array(fives)

	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.scatter(fives[:,0], fives[:,1], c='b', label='Features of digit 5')
	plt.scatter(ones[:,0], ones[:,1], c='y', label='Features of digit 1')
	plt.legend()
	plt.savefig('train_features_img' + str(img_index) + '.png')
	plt.gcf().clear()
	img_index += 1

	### END YOUR CODE

def visualize_result(X, y, W):
	'''This function is used to plot the linear model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_samples,].
	
	Returns:
		No return. Save the plot to 'train_result.*' and include it
		in submission.
	'''
	### YOUR CODE HERE

	global img_index
	slope = -(W[0]/W[2])/(W[0]/W[1])
	intercept = -W[0]/W[2]
	boundary = []
	x_vals = np.linspace(-1,0.2,num=2)
	for i in x_vals:
		boundary.append((slope*i) + intercept)
	plt.plot(x_vals, boundary, c='r', label='Decision Boundary')
	ones = []
	fives = []
	for i in range(y.shape[0]):
		if y[i]==1:
			ones.append(X[i])
		else:
			fives.append(X[i])
	ones = np.array(ones)
	fives = np.array(fives)
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.scatter(fives[:,0], fives[:,1], c='b', label='Features of digit 5')
	plt.scatter(ones[:,0], ones[:,1], c='y', label='Features of digit 1')
	plt.legend()
	plt.savefig('test_features_img' + str(img_index) + '.png')
	plt.gcf().clear()
	img_index += 1

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
	global img_index
	raw_data = load_data(os.path.join(data_dir, train_filename))
	raw_train, raw_valid = train_valid_split(raw_data, 1000)
	train_X, train_y = prepare_data(raw_train)
	valid_X, valid_y = prepare_data(raw_valid)

	# Visualize training data.
	visualize_features(train_X[:, 1:3], train_y)

	# ------------Perceptron------------
	perceptron_models = []
	min_validation_error = 1000
	best_perceptron = None
	for max_iter in [10, 20, 50, 100, 200]:
		# Initialize the model.
		perceptron_classifier = perceptron(max_iter=max_iter)
		perceptron_models.append(perceptron_classifier)

		# Train the model.
		perceptron_classifier.fit(train_X, train_y)
		
		print('Max interation:', max_iter)
		print('Weights after training:',perceptron_classifier.get_params())
		print('Training accuracy(in %):', perceptron_classifier.score(train_X, train_y))
		print('Validation accuracy(in %):', perceptron_classifier.score(valid_X, valid_y))
		print()

		if (min_validation_error > perceptron_classifier.score(valid_X, valid_y)):
			min_validation_error = perceptron_classifier.score(valid_X, valid_y)
			best_perceptron = perceptron_classifier

	# Visualize the the 'best' one of the five models above after training.
	### YOUR CODE HERE

	visualize_result(train_X[:,1:3], train_y, best_perceptron.get_params())

	### END YOUR CODE
	
	# Use the 'best' model above to do testing.
	### YOUR CODE HERE

	raw_test_data = load_data(os.path.join(data_dir, test_filename))
	test_X, test_y = prepare_data(raw_test_data)
	print('Testing accuracy(in %):', perceptron_classifier.score(test_X, test_y))
	print()

	# ### END YOUR CODE


	# ------------Logistic Regression------------

	# Check GD, SGD, BGD
	logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

	logisticR_classifier.fit_GD(train_X, train_y)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))
	print()
	# plt.plot(logisticR_classifier.errors)
	# plt.show()

	logisticR_classifier.fit_BGD(train_X, train_y, 1000)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))
	print()
	# plt.plot(logisticR_classifier.errors)
	# plt.show()

	logisticR_classifier.fit_SGD(train_X, train_y)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))
	print()
	# plt.plot(logisticR_classifier.errors)
	# plt.show()

	logisticR_classifier.fit_BGD(train_X, train_y, 1)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))
	print()
	# plt.plot(logisticR_classifier.errors)
	# plt.show()

	logisticR_classifier.fit_BGD(train_X, train_y, 10)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))
	print()
	# plt.plot(logisticR_classifier.errors)
	# plt.show()

	# Explore different hyper-parameters.
	## YOUR CODE HERE

	print('Tuning hyper-parameters...')
	errors = []
	for learning_rate in np.linspace(0,10,num=100):
		logisticR_classifier = logistic_regression(learning_rate, max_iter=100)
		logisticR_classifier.fit_BGD(train_X, train_y, 10)
		accuracy = logisticR_classifier.score(valid_X, valid_y)
		errors.append(100-accuracy)
	plt.xlabel('Learning Rate')
	plt.ylabel('Error')
	plt.plot(errors)
	plt.savefig('variation_learning_rate_img' + str(img_index) + '.png')
	plt.gcf().clear()
	img_index += 1

	errors = []
	for iter in range(0,1000,100):
		logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=iter)
		logisticR_classifier.fit_BGD(train_X, train_y, 10)
		accuracy = logisticR_classifier.score(valid_X, valid_y)
		errors.append(100-accuracy)
	plt.plot(errors)
	plt.xlabel('Iterations')
	plt.ylabel('Error')
	plt.savefig('variation_max_iters_img' + str(img_index) + '.png')
	plt.gcf().clear()
	img_index += 1

	best_logisticR = logistic_regression(learning_rate=0.5, max_iter=100)
	best_logisticR.fit_GD(train_X, train_y)

	## END YOUR CODE

	# Visualize the your 'best' model after training.
	## YOUR CODE HERE

	visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

	## END YOUR CODE

	# Use the 'best' model above to do testing.
	## YOUR CODE HERE

	raw_test_data = load_data(os.path.join(data_dir, test_filename))
	test_X, test_y = prepare_data(raw_test_data)
	print('Testing accuracy(in %):', best_logisticR.score(test_X, test_y))
	print()

	## END YOUR CODE

if __name__ == '__main__':
	main()