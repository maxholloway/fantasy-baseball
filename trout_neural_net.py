import csv
import numpy as np

########################### Data Normalization ############################
def normalize(file_name):
	'''
	inputs: the name of the file (in the local directory) which has all features and outputs
	outputs: a 2D array with all of the normalized features and outputs
	how it works: basically, every column corresponds to a different feature;
								subtracting the minimum and dividing by the range of that column
								on every element of that column scales every feature to between
								0 and 1.
	runtime: runs in O(N) time and space, where N is proportional to the
					 number of entries in the csv file.
	'''
	
	## Functions that help with normalization ##
	def non_modular_date(date):
		'''
		inputs: the string from retrosheet.org with the date
		outputs: a number that represents the (approximate) number of days since 2000
		'''
		month = int(date[:2])
		day = int(date[3:5])
		year = int(date[6:10])
		
		return day + 30.5*(month + 12*year)

	def column_maximums(array):
		'''
		inputs: a 2D list/array of dimension MxN of numerical data
		outputs: a 1D list of dimension 1xN with the maximum of the N columns
		'''
		assert(len(array) >=1)
		M = len(array)
		N = len(array[0])
		
		maximums = [0]*N
		for i in range(M):
			assert(len(array[i]) == N)
			for j in range(N):
				if(array[i][j] > maximums[j]):
					maximums[j] = array[i][j]
		
		return maximums

	def column_minimums(array):
		'''
		inputs: a 2D list/array of dimension MxN of numerical data
		outputs: a 1D list of dimension 1xN with the minimum values of the N columns
		'''
		assert(len(array) >=1)
		M = len(array)
		N = len(array[0])
		
		minimums = column_maximums(array) # known to be greater than the true minimums
		for i in range(M):
			assert(len(array[i]) == N)
			for j in range(N):
				if(array[i][j] < minimums[j]):
					minimums[j] = array[i][j]
		
		return minimums

	def column_ranges(array):
		'''
		inputs: a 2D list/array of dimension MxN of numerical data
		outputs: a 1D list of dimension 1xN with the ranges of the elements in the N columns
		'''
		maxes, mins = column_maximums(array), column_minimums(array)
		assert(len(maxes) == len(mins))
		return [maxes[i] - mins[i] for i in range(len(maxes))]

	def elementwise_subtraction(list1, list2):
		assert(len(list1) == len(list2))
		return [list1[i]-list2[i] for i in range(len(list1))]
		
	def elementwise_division(list1, list2):
		'''
		Performs elementwise division of list1 / list2
		'''
		assert(len(list1) == len(list2))
		assert(0 not in list2) # prevents division by 0
		return [list1[i]/list2[i] for i in range(len(list1))]
		

	player_data = list(csv.reader(open(file_name)))

	# Takes into account only the rows that includes pitcher data
	with_pitcher = []
	for row in player_data:
		if(len(row) == 28):
			this_row = []
			for j in range(len(row)):
				if(j == 0):
					this_row += [non_modular_date(row[j])]
				else:
					this_row += [float(row[j])]
			with_pitcher += [this_row]

	minimums = column_minimums(with_pitcher)
	ranges = column_ranges(with_pitcher)

	# Subtract the minimum of the column and divide by the maximum of the column
	# for each entry, such that they are in [0, 1].
	min_subtracted = [elementwise_subtraction(row, minimums) for row in with_pitcher]
	range_divided = [elementwise_division(row, ranges) for row in min_subtracted]
	return range_divided

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
	TRAINING_PROPORTION = .8 # the proportion of the data used for training
		
	normalized_data = np.array(normalize(file_name))
	np.random.shuffle(normalized_data) # shuffle the rows randomly

	num_training = int(TRAINING_PROPORTION*(normalized_data.shape[0])) # number of training rows
	num_test = normalized_data.shape[0]-num_training # number of test rows
	training_data, test_data = normalized_data[:num_training+1], normalized_data[num_training+1:]
	training_features, test_features = training_data[:, [i for i in range(27)]], test_data[:, [i for i in range(27)]]
	training_labels, test_labels = training_data[:, [27]], test_data[:, [27]]
	# print('Original data, split into training and test:')
	# print(training_data.shape)
	# print(test_data.shape)
	# print('\nFeatures:')
	# print(training_features.shape)
	# print(test_features.shape)
	# print('\nLabels')
	# print(training_labels.shape)
	# print(test_labels.shape)
	
	
if __name__ == '__main__':
	make_neural_net('Mike Trout.csv')