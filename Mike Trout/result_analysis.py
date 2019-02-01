if __name__ == '__main__':
	import pandas as pd

	df = pd.read_csv("Mike Trout.csv", dtype=str) # make a dataframe of Mike Trout's data
	df = df.dropna()
	df['Draftkings Score'] = pd.to_numeric(df['Draftkings Score'])
	max_score = df['Draftkings Score'].max()
	min_score = df['Draftkings Score'].min()
	range_scores = max_score-min_score


	scaled_scores = [(score - min_score)/range_scores for score in df['Draftkings Score']]
	scores = [score for score in scaled_scores]

	mean_score = sum(scores)/len(scores)
	print('Mike Trout\'s mean scaled score is', str(mean_score) + '.\n')

	se = 0
	for score in scaled_scores: se += (score-mean_score) ** 2
	mse = se/len(scaled_scores)

	print('If we guess that Mike Trout will perform average every time,\nthen the mean-squared error is', str(mse)+'.\n')