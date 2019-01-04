# fantasy-baseball
Holds all of the files related to Anthony and Maxwell Holloway's fantasy 
baseball prediction project.

The basic idea of this project is simple. My father and I thought that 
there would be potential for machine learning to predict the performance of a 
player on a day-to-day basis. Standard guessing methods may estimate a 
player's worth by taking his average performance over the season, 
however this is missing all of the nuance of their performance based on 
recorded factors, such as the pitcher, the time of year, the weather,
the location (home vs away), and others.

In an effort to make educated guesses on the performance of the 
guessing, we developed a simple model.

1.) Train a neural net for every batter, using a training set taken from
    every game they've played (using both the Lahman dataset and 
    retrosheet.org).
2.) Use a solving tool in excel to find which combination of players
    will maximize the number of points scored while remaining under the
    salary cap of the service.

Although this is a simple process, it will hopefully provide at least 
marginally better results than computing the average.
