import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

######################### Creating the Neural Net #########################
def make_neural_net(file_name):
	'''
	inputs: the name of the csv file with the player's data
	outputs: a neural net object that is trained with the 80% of the data
	how it works: basically follow regular protocol for making a neural net
								using tensorflow. This method is certainly not set in stone,
								and may be modified for different types of players or for
								different years or whatever.
	runtime: long :))
	'''
	
	TESTING_PROPORTION = .2 # the proportion of the data used for training
	
	# Loads data into the training and test sets necessary
	def non_modular_date(date):
		'''
		inputs: the string from retrosheet.org with the date
		outputs: a number that represents the (approximate) number of days since 2000
		'''
		month = int(date[:2])
		day = int(date[3:5])
		year = int(date[6:10])

		return day + 30.5 * (month + 12 * year)

	df = pd.read_csv("Mike Trout.csv", dtype=str) # make a dataframe of Mike Trout's data
	df['Date'] = df['Date'].apply(non_modular_date); # convert the dates from modular to regular
	df.astype('float32')

	#### NOT MY CODE ####
	training_data_df, test_data_df = train_test_split(df, test_size=TESTING_PROPORTION)

	# Pull out columns for X (data to train with) and Y (value to predict)
	X_training = training_data_df.drop('Draftkings Score', axis=1).values
	Y_training = training_data_df[['Draftkings Score']].values

	# Pull out columns for X (data to train with) and Y (value to predict)
	X_testing = test_data_df.drop('Draftkings Score', axis=1).values
	Y_testing = test_data_df[['Draftkings Score']].values

	# All data needs to be scaled to a small range like 0 to 1 for the neural
	# network to work well. Create scalers for the inputs and outputs.
	X_scaler = MinMaxScaler(feature_range=(0, 1))
	Y_scaler = MinMaxScaler(feature_range=(0, 1))

	# Scale both the training inputs and outputs
	X_scaled_training = X_scaler.fit_transform(X_training)
	Y_scaled_training = Y_scaler.fit_transform(Y_training)

	# It's very important that the training and test data are scaled with the same scaler.
	X_scaled_testing = X_scaler.transform(X_testing)
	Y_scaled_testing = Y_scaler.transform(Y_testing)

	# Define model parameters
	learning_rate = 0.001
	training_epochs = 100
	display_step = 5

	# Define how many inputs and outputs are in our neural network
	number_of_inputs = 27
	number_of_outputs = 1

	# Define how many neurons we want in each layer of our neural network
	layer_1_nodes = 50
	layer_2_nodes = 100
	layer_3_nodes = 50

	# Section One: Define the layers of the neural network itself

	# Input Layer
	with tf.variable_scope('input'):
		X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

	# Layer 1
	with tf.variable_scope('layer_1'):
		weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
		layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

	# Layer 2
	with tf.variable_scope('layer_2'):
		weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
		layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

	# Layer 3
	with tf.variable_scope('layer_3'):
		weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
		layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

	# Output Layer
	with tf.variable_scope('output'):
		weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
		prediction = tf.matmul(layer_3_output, weights) + biases

	# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

	with tf.variable_scope('cost'):
		Y = tf.placeholder(tf.float32, shape=(None, 1))
		cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

	# Section Three: Define the optimizer function that will be run to optimize the neural network

	with tf.variable_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	# Initialize a session so that we can run TensorFlow operations
	with tf.Session() as session:
		# Run the global variable initializer to initialize all variables and layers of the neural network
		session.run(tf.global_variables_initializer())

		# Run the optimizer over and over to train the network.
		# One epoch is one full run through the training data set.
		for epoch in range(training_epochs):

			if (epoch % 5 == 0):
				training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
				testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
				print(epoch, training_cost, testing_cost)

			# Feed in the training data and do one step of neural network training
			session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

			# Print the current training status to the screen
			print("Training pass: {}".format(epoch))

		# Training is now complete!
		print("Training is complete!")
		final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
		final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
		print("Training: ", final_training_cost, "\nTesting: ", final_testing_cost)
#### END OF NOT MY CODE ####



if __name__ == '__main__':
	make_neural_net('Mike Trout.csv')
