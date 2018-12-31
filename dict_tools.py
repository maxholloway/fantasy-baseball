def reverse_dict(d):
	'''
	inputs: a dictionary (d)
	outputs: another dictionary where the keys are the values
	runtime: O(N) time and space, where N is the length of the dictionary
	'''
	newDict = {}
	for key, val in d.items():
		newDict[val] = key
	return newDict
	
if __name__ == '__main__':
	d1 = {'a':'alpha', 'b':'beta', 'g':'gamma', 'z':'omega'}
	print(d1)
	print(reverse_dict(d1))