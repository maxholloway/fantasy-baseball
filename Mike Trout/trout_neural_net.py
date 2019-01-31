import csv
import numpy as np
import tensorflow as tf
import pandas as pd

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

	def index_of_label(score, num_labels):
		'''
		inputs: 
			score: the score of the player; ranges between [0, 1]
			num_labels: this is how many lables there are total
		outputs: a 1D numpy array of length n=num_labels, with either 0s in all spots except
						 for one 1; this 1 will indicate the category of this label
		'''
		assert num_labels > 0
		interval_size = 1/num_labels # the range that each label sweeps over; made constant for simplicity
		return min([int(score/interval_size), num_labels-1])

		
	TRAINING_PROPORTION = .8 # the proportion of the data used for training
	
	# Loads data into the training and test sets necessary
	normalized_data = np.array(normalize(file_name))
	np.random.shuffle(normalized_data) # shuffle the rows randomly

	num_training = int(TRAINING_PROPORTION*(normalized_data.shape[0])) # number of training rows
	num_test = normalized_data.shape[0]-num_training # number of test rows
	training_data, test_data = normalized_data[:num_training+1], normalized_data[num_training+1:]
	training_features, test_features = training_data[:, [i for i in range(27)]], test_data[:, [i for i in range(27)]]

	training_labels, test_labels = training_data[:, 27], test_data[:, 27]

	training_labels = np.array(training_labels)
	test_labels = np.array(test_labels)

if __name__ == '__main__':
	make_neural_net('Mike Trout.csv')
