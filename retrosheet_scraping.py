from bs4 import BeautifulSoup
import requests
from csv import writer

def generateLinks(player_name):
	'''
	inputs: the name of the player (as a string)
	outputs: a list of links to the game-by-game retrosheets for that player
	STILL MUST WRITE THIS!!
	'''
	pass
	
def retrosheet_to_csv(player_name, urls):
	'''
	inputs:
					player_name: the name of the player that the data will be on (string)
					urls: a list of urls that give the years the player played (list of strings)
	outputs: no output to command line; it makes a .csv file with the name of the player,
					 though
	how it works: for every url, it scrapes the player's data from that url and converts
								it to .csv format
	'''
	for i in range(len(urls)):
		url = urls[i]
		response = requests.get(url)
		soup = BeautifulSoup(response.text, 'html.parser')
		table_from_retrosheet = soup.find_all('pre')[4] # the way the website is set up, this is the 5th 'pre'
		text_table = table_from_retrosheet.get_text().split('\n') # makes a table where the rows are each an element of the list


		## ---- Making the header names for the csv file ---- ##
		header_names = text_table[0].split() # headers for csv
		header_names[1] = 'Home' # 1 if home, 0 if away
		header_names.pop(3)

		## ---- Making a list of game strings for the csv file ---- ##
		games = text_table[1:] # all of the rows after the headers; needs cleaned up
		# Cleaning up rows; gets rid of the useless rows that reiterate the headers
		# or that are empty, because these rows do not represent actual games
		games = filter((lambda x: ('Date' not in x) and (x.split() != [])), games)

		## ---- Setting the mode for the csv writing or appending ---- ##
		if(i == 0):
			mode = 'w' # if this is the first year of games, then make a new csv file
		else:
			mode = 'a' # if this isn't the first year of games, then append to the csv file

		## ---- Adding game data to the csv file ---- ##
		with open(player_name + '.csv', mode) as csv_file:
			csv_writer = writer(csv_file)
			
			if(i == 0): csv_writer.writerow(header_names) # only write header names once
		
			for game in games:
				game_data = []
				
				date = game[:10]
				game_data += [date]
				
				game = game[21:]
				home = 1 if game[:2] == 'VS' else (0 if game[:2] == 'AT' else 'fucked up')
				game_data += [home]
				
				game = game[3:]
				
				rest_of_data = game.split()
				rest_of_data.pop(1)
				rest_of_data.pop(1)

				game_data += rest_of_data
				csv_writer.writerow(game_data)

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
									'https://www.retrosheet.org/boxesetc/2012/Itroum0010022012.htm']
	name_for_trout = 'Mike Trout'
	retrosheet_to_csv(name_for_trout, urlsForTrout)