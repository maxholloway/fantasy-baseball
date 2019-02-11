import keras
import random
from csv import writer
import numpy as np
import helper_functions as hf

######################### Creating the Neural Net #########################
def make_neural_net(file_name, learn_rate, training_epochs, layers, model_name, drop_rate=.3, testing_proportion=.1):
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


  df = hf.make_dataframe(file_name)

 # Scale both the testing and training inputs and outputs
  X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing = hf.make_train_test(df, testing_proportion)


  # Constructing the neural network's layers/defining the model
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(layers[1], input_dim=layers[0], activation='relu'))
  for i in range(2, len(layers)-1):
    model.add(keras.layers.Dense(layers[i], activation='relu'))
    model.add(keras.layers.Dropout(drop_rate))
  model.add(keras.layers.Dense(layers[-1], activation='linear'))
  model.compile(optimizer='adam', loss='mse') # because trying to predict score, gives a loss function of MSE

  # Training the model
  model.fit(X_scaled_training,
            Y_scaled_training,
            epochs=training_epochs,
            shuffle=True,
            verbose=0)

  std_error_rate = 0.03555387959152568
  train_error_rate = model.evaluate(X_scaled_training, Y_scaled_training, verbose=0)
  test_error_rate = model.evaluate(X_scaled_testing, Y_scaled_testing, verbose=0)
  print('\n\nThe train MSE is {}.\n'.format(train_error_rate))
  print('The test MSE is {}.\n'.format(test_error_rate))
  print('Succeeded by {} on training and {} on testing\n\n'.format(std_error_rate-train_error_rate, std_error_rate-test_error_rate))

  model.save('Models/'+model_name)

  with open('Log_keras.csv', 'a') as csv_file:
    writer(csv_file).writerow(
      [layers, learn_rate, training_epochs, train_error_rate, test_error_rate, std_error_rate])


if __name__ == '__main__':
  for i in range(500):
    net_name = 'Neural net #{}.h5'.format(i+1)
    num_epochs = random.randint(75, 100)
    layers = [random.randint(20, 100) for j in range(7)]  # random number of nodes in each layer
    dropout_rate = .2 + .4*np.random.random() # range of dropout rate: [.2, .6)
    make_neural_net('Mike Trout.csv', .001, num_epochs, [31] + layers + [1],
                    net_name, drop_rate=dropout_rate, testing_proportion=.1)
