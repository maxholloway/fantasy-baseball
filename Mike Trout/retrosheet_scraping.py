from bs4 import BeautifulSoup
import requests
from csv import writer, reader

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

abbreviation_to_team = {'ANA': 'Anaheim Angels', 'ARI': 'Arizona Diamondbacks',
												'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
												'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs',
												'CHI': 'Chicago White Sox', 'CIN': 'Cincinnati Reds',
												'CLE': 'Cleveland Indians', 'COL': 'Colorado Rockies',
												'DET': 'Detroit Tigers', 'FLA': 'Florida Marlins', 
												'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
												'LAD': 'Los Angeles Dodgers', 'LA': 'Los Angeles Dodgers',
												'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins',
												'NYM': 'New York Mets', 'NY': 'New York Yankees',
												'OAK': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
												'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres',
												'SEA': 'Seattle Mariners', 'SF': 'San Francisco Giants',
												'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
												'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays',
												'WSH': 'Washington Nationals', 'WAS': 'Washington Nationals'}

lahman_abbrev_to_team = {'LAA': 'Anaheim Angels', 'ARI': 'Arizona Diamondbacks',
												'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
												'BOS': 'Boston Red Sox', 'CHN': 'Chicago Cubs',
												'CHA': 'Chicago White Sox', 'CIN': 'Cincinnati Reds',
												'CLE': 'Cleveland Indians', 'COL': 'Colorado Rockies',
												'DET': 'Detroit Tigers', 'FLO': 'Florida Marlins', 
												'HOU': 'Houston Astros', 'KCA': 'Kansas City Royals',
												'LAN': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
												'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins',
												'NYN': 'New York Mets', 'NYA': 'New York Yankees',
												'OAK': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
												'PIT': 'Pittsburgh Pirates', 'SDN': 'San Diego Padres',
												'SEA': 'Seattle Mariners', 'SFN': 'San Francisco Giants',
												'SLN': 'St. Louis Cardinals', 'TBA': 'Tampa Bay Rays',
												'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays',
												'WAS': 'Washington Nationals'}



def generateLinks(player_name):
	'''
	inputs: the name of the player (as a string)
	outputs: a list of links to the game-by-game retrosheets for that player
	STILL MUST WRITE THIS!!
	'''
	pass
	
def dot(list1, list2):
	'''
	gives the dot product of two lists, (element-wise product)
	'''
	assert(len(list1) == len(list2)) # should be same length
	return sum([list1[i]*list2[i] for i in range(len(list1))])
	
def stats_to_draftkings_score(stats, player_type):
	'''
	inputs: 
				stats is a list with the number of times the following occur, respectively:
					[Single, Double, Triple, Home Run, Run Batted In, Run, Base on Balls,
					Hit By Pitch, Stolen Base].. if the player is a pitcher, then the
					vector is [Inning pitched, Strikeout, Win, Earned Run Allowed, 
					Hit Against, Base on Balls Against, Hit Batsman, Complete Game, 
					Complete Game Shutout, No Hitter];
					
				player_type is a string that can either be 'hitter' or 'pitcher',
					and they have different ways of scoring.
	outputs: the score that the hitter would get
	'''
	assert(player_type in {'hitter', 'pitcher'})
	if(player_type == 'hitter'):
		hitter_vec = [3, 5, 8, 10, 2, 2, 2, 2, 5]
		return dot(hitter_vec, stats)
	else:
		pitcher_vec = [2.25, 2, 4, -2, -.6, -.6, -.6, 2.5, 2.5, 5]
		return dot(pitcher_vec, stats)
		
def batter_to_csv(player_name, urls):
	'''
	inputs: the name of the player and the urls to that player's years
	outputs: a csv file with the name of the player; the headers of the csv
					 file will be all of the n features from [0, n), then the draftkings
					 score that the player received for that day. The n features can
					 be used as the inputs to a neural net, and the draftkings score
					 can be used as the output of the neural net.
	'''
	
	pitcher_data = list(reader(open('pitching_2005-2017.csv')))

	def is_prefix(alleged_prefix, word):
		'''
		inputs: an alleged prefix to a word, both in string format
		outputs: True if the alleged prefix is indeed a prefix; false otherwise
		'''
		return alleged_prefix == word[:len(alleged_prefix)]

	def game_to_pitcher_last_name(url_of_game, pitcher_team):
		'''
		inputs: the url to the game
		outputs: the name of the starting pitcher for the opposing team
		'''
		
		last_name = 'PITCHER NOT FOUND' # if the pitcher isn't found, this value is never changed
		game_page = BeautifulSoup(requests.get(url_of_game).text, 'html.parser')
		pres = game_page.find_all('pre')
		for i in range(len(pres)):
			if(pres[i].get_text() == 'PITCHING'):
				for k in range(i+1, len(pres)):
					lines = pres[k].get_text().split('\n')
					header_line = lines[0]
					if(is_prefix(pitcher_team, header_line)):
						last_name = lines[1].split()[0]
						break
				break
		return last_name
		
	def get_pitcher_data(pitcher_last_name, pitcher_team_name, pitcher_year):
		'''
		inputs: self explanatory; they're all strings, too
		outputs: a row with the specified pitcher's data for the specified year
						 and team
		how it works: basically it goes through a csv file and finds the pitcherf
		'''
		# Make all lowercase for the .csv file format
		pitcher_last_name = pitcher_last_name.lower()
		# Make copy of pitcher data
		new_pitcher_data = pitcher_data[1:]
		
		# Filter by the year
		new_pitcher_data = list(filter((lambda x: int(x[1]) == pitcher_year), new_pitcher_data))

		# Filter by last name
		prefix_last_name = pitcher_last_name[:5]

		new_pitcher_data = list(filter((lambda x: is_prefix(prefix_last_name, x[0])), new_pitcher_data))
		# Filter by team name
		lahman_team_to_abbrev = reverse_dict(lahman_abbrev_to_team)
		team_abbrev = lahman_team_to_abbrev[pitcher_team_name]
		
		new_pitcher_data = list(filter((lambda x: team_abbrev == x[3]), new_pitcher_data))
		
		if(len(list(new_pitcher_data)) == 1): # asserting that there's only one pitcher on this team, this year, of this name
			return new_pitcher_data[0]
		else:
			return None 
	
	for i in range(len(urls)):
		response = requests.get(urls[i])
		soup = BeautifulSoup(response.text, 'html.parser')
		table_from_retrosheet = soup.find_all('pre')[4] # the way the website is set up, this is the 5th 'pre'
		text_table = table_from_retrosheet.get_text().split('\n') # makes a table where the rows are each an element of the list		
		
		year = int(soup.find('h2').get_text()[4:8]) # the year this url is related to
		if(year < 2005): continue
		
		''' Getting the links to the websites for the game; this can lead to 
				information regarding the opposing pitcher '''
		anchors = table_from_retrosheet.find_all('a')
		game_links = []
		for j in range(len(anchors)):
			game_links += ['https://www.retrosheet.org/boxesetc' + anchors[j].get('href')[2:]] if j%2 == 1 else []
			
		## ---- Making the header names for the csv file ---- ##
		if(i == 0):
			column_titles = text_table[0].split() # headers for csv
			column_titles[1] = 'Home' # 1 if home, 0 if away
			column_titles += ['Draftkings Score']
			# print('Header names: ', column_titles)
			# print('length of header names: ', len(column_titles))
			with open(player_name + '.csv', 'w') as csv_file: writer(csv_file).writerow(column_titles)

		## ---- Making a list of game strings for the csv file ---- ##
		games = text_table[1:] # all of the rows after the headers; needs cleaned up
		# Cleaning up rows; gets rid of the useless rows that reiterate the headers
		# or that are empty, because these rows do not represent actual games
		games = list(filter((lambda x: ('Date' not in x) and (x.split() != [])), games))

		with open(player_name + '.csv', 'a') as csv_file:
			csv_writer = writer(csv_file)
			
			for j in range(len(games)):
				game = games[j] # the batting info for the game
				game_link = game_links[j] # link to the general game info
				
				features, output = [], [] # features should have home, date, batting position, and pitcher data

				## Getting gameday features, including pitcher data
				### Date that the game was played
				date = game[:10]
				
				### Whether the game was at home or away
				game = game[21:]
				home = 1 if game[:2] == 'VS' else (0 if game[:2] == 'AT' else 'fucked up')
				game = game[3:]
				### This player's batting position 
				rest_of_data = game.split()
				batting_position = rest_of_data[-2]
				features += [date, home, batting_position]
				## End of 'Getting gameday features'
				
				# Getting data on how the pitcher performed that same year
				opposing_team_abbrev = rest_of_data[0]
				opposing_team = abbreviation_to_team[opposing_team_abbrev]
				pitcher_last_name = game_to_pitcher_last_name(game_link, opposing_team)

				pitcher =  get_pitcher_data(pitcher_last_name, opposing_team, year)
				if(pitcher != None):
					if(int(pitcher[6]) > 0):
						win_loss_ratio = int(pitcher[5])/int(pitcher[6])
					else:
						win_loss_ratio = 0
					rest_of_pitcher_data = pitcher[7:]
					relevant_pitcher_data = [win_loss_ratio] + rest_of_pitcher_data
					features += relevant_pitcher_data
				# End of 'Getting features'
				
				# Making output				
				if(rest_of_data[-1] != 'p'):
					at_bats, runs, hits, doubles, triples, home_runs, rbis, bb, ibb, so, hbp = rest_of_data[3:14]
					singles = int(hits) - (int(doubles) + int(triples) + int(home_runs))
					assert(singles >= 0)
					stolen_bases = rest_of_data[-7]
					batter_data = [singles, doubles, triples, home_runs, rbis, runs, bb, hbp, stolen_bases]
					batter_data = list(map((lambda x: int(x)), batter_data)) # convert from string to int
					score = stats_to_draftkings_score(batter_data, 'hitter')
					output = [score]
				# End of making output
				
				# append features and output, in that order, to the csv file
				csv_writer.writerow(features + output)

		
		
def make_csv_files(player_names):
	'''
	inputs: the names of players that should have CSV files made for them
	outputs: none, other than a CSV file for each player
	'''
	
	for player_name in player_names:
		links_to_players_sheets = generateLinks(player_name)
		retrosheet_to_csv(player_name, links_to_players_sheets)

if __name__ == '__main__':
	urlsForTrout = ['https://www.retrosheet.org/boxesetc/2011/Itroum0010012011.htm',
									'https://www.retrosheet.org/boxesetc/2012/Itroum0010022012.htm',
									'https://www.retrosheet.org/boxesetc/2013/Itroum0010032013.htm',
									'https://www.retrosheet.org/boxesetc/2014/Itroum0010042014.htm',
									'https://www.retrosheet.org/boxesetc/2015/Itroum0010052015.htm',
									'https://www.retrosheet.org/boxesetc/2016/Itroum0010062016.htm',
									'https://www.retrosheet.org/boxesetc/2017/Itroum0010072017.htm']
									# 'https://www.retrosheet.org/boxesetc/2018/Itroum0010082018.htm'] # does not include pitching data
	name_for_trout = 'Mike Trout'
	batter_to_csv(name_for_trout, urlsForTrout)
