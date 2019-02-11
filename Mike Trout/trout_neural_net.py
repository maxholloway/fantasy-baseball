import random

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from csv import writer, reader

######################### Creating the Neural Net #########################
def make_neural_net(file_name, learning_rate, training_epochs, layers):
	'''
	inputs: the name of the csv file with the player's data
	outputs: a neural net object that is trained with the 80% of the data
	how it works: basically follow regular protocol for making a neural net
								using tensorflow. This method is certainly not set in stone,
								and may be modified for different types of players or for
								different years or whatever.
	runtime: long :))
	'''

	assert len(layers) == 9

	TESTING_PROPORTION = .1 # the proportion of the data used for training
	
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
	df = df.dropna() # remove all rows with nan's
	df['Date'] = df['Date'].apply(non_modular_date); # convert the dates from modular to regular

	# Converts all values to numerics
	for col_name in df.columns.values:
		df[col_name] = pd.to_numeric(df[col_name])

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


	# Defines number of inputs and number of nodes in all of the layers
	number_of_inputs, layer_1_nodes, layer_2_nodes, layer_3_nodes,\
	layer_4_nodes, layer_5_nodes, layer_6_nodes, layer_7_nodes, number_of_outputs = layers


	# Section One: Define the layers of the neural network itself

	# Input Layer
	with tf.variable_scope('input'):
		X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

	# Layer 1
	with tf.variable_scope('layer_1', reuse=False):
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

	# Layer 4
	with tf.variable_scope('layer_4'):
		weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, layer_4_nodes],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases4", shape=[layer_4_nodes], initializer=tf.zeros_initializer())
		layer_4_output = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

	# Layer 5
	with tf.variable_scope('layer_5'):
		weights = tf.get_variable(name="weights5", shape=[layer_4_nodes, layer_5_nodes],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases5", shape=[layer_5_nodes], initializer=tf.zeros_initializer())
		layer_5_output = tf.nn.relu(tf.matmul(layer_4_output, weights) + biases)

	# Layer 6
	with tf.variable_scope('layer_6'):
		weights = tf.get_variable(name="weights6", shape=[layer_5_nodes, layer_6_nodes],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases6", shape=[layer_6_nodes], initializer=tf.zeros_initializer())
		layer_6_output = tf.nn.relu(tf.matmul(layer_5_output, weights) + biases)

	# Layer 7
	with tf.variable_scope('layer_7'):
		weights = tf.get_variable(name="weights7", shape=[layer_6_nodes, layer_7_nodes],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases7", shape=[layer_7_nodes], initializer=tf.zeros_initializer())
		layer_7_output = tf.nn.relu(tf.matmul(layer_6_output, weights) + biases)

	# Output Layer
	with tf.variable_scope('output'):
		weights = tf.get_variable(name="weights8", shape=[layer_7_nodes, number_of_outputs],
															initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable(name="biases8", shape=[number_of_outputs], initializer=tf.zeros_initializer())
		prediction = tf.matmul(layer_7_output, weights) + biases

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

			# if (epoch % 5 == 0):
			# 	training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
			# 	testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
			# 	print(epoch, training_cost, testing_cost)

			# Feed in the training data and do one step of neural network training
			session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

			# Print the current training status to the screen
			# print("Training pass: {}".format(epoch))

		# Training is now complete!
		#### END OF NOT MY CODE ####

		print("Training is complete!")
		final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
		final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
		print("Training: ", final_training_cost, "\nTesting: ", final_testing_cost)
		print("Succeeded by: ", 0.03555387959152568 - final_testing_cost)

		with open('Log.csv', 'a') as csv_file:
			writer(csv_file).writerow([layers, learning_rate, training_epochs, final_training_cost, final_testing_cost, 0.03555387959152568])


if __name__ == '__main__':
	for i in range(1):
		layers = [random.randint(20, 100) for j in range(7)] # random number of nodes in each layer
		# layers = [48, 52, 34, 75, 83, 56, 20]
		# num_epochs = random.randint(35, 75)
		num_epochs = 50
		make_neural_net('Mike Trout.csv', .001, num_epochs, [31] + layers + [1])
