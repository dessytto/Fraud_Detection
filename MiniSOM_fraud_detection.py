# Self Organizing Map for fraud detection from customers' applications
# Goal: Identify patterns int he high-dim dataset to predict fraud likelyhood
# The dataset is from UCI Machine Learning Repository. It is "Statlog (Aus Credit Approval) Data Set".
# All atribute names have been changed to meaningless symbols for confidentiality,  hence we need a deep learning algorithm.
# Each line is a customer. We want to identify various segments, and the cheaters should stand out as outliers.
# MID: mean interneuron distance. Outliers will be far from the other neurons in their neighborhood.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# Split the dataset into 2: The last column ('Class') should be separated.
# Class = 1 means application was approved. This is NOT done for the purposes we do it in SUPERVISED ML.
X = dataset.iloc[:,:-1].values #everything but the last column
y = dataset.iloc[:, -1].values #this is the last column only, the only dependent variable

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#Training the SOM (we will not implement it from scratch, but use SO's implementation: MiniSom 1.0 - numpy based)
from minisom import MiniSom
# x and y are the dimentions of the SOM grid. Input_len is the # of features.
# sigma - spread of the neighborhood function (Gaussian)
# A decay function could be used to improve the convergence rate.
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# Initialize the weights
som.random_weights_init(X)

# Train the SOM on X
som.train_random(data = X, num_iteration = 100)

#Visualize the results
#MID = mean of the distances to all the neighb. neurons inside a neighborhood defined via the radius sigma.
#Large MID = white.
from pylab import bone, pcolor, colorbar, plot, show
#Initialize the figure:
bone()
#som.distance_map() returns all the MIDs onto a matrix, and we will transpose it
#Put the winning nodes into a map, color by MID:
pcolor(som.distance_map().T)
#add colorbar
colorbar()

#The cheaters have been identified as the color(white) outliers. Mark the customers who got approval (y dataset):
# red circles (not approved) and green squares (approved)
markers = ['*','s']
colors = ['r','g']
#loop over all customers. i loops over customer indices, x the vectors of each customer:
for i, x in enumerate(X):
    w = som.winner(x) #this gets us the winning node of customer x
    plot(w[0] + 0.5, # w[i] are the bottom left coordinates of each square
         w[1] + 0.5, #on this winning node, we plot the marker at the center of the square
         markers[y[i]], #i is the index of the customer, and y is the dep. var. we are marking and coloring by
         markeredgecolor = colors[y[i]], #color only the edge
         markerfacecolor = "None",
         markersize = 15,
         markeredgewidth = 2)
show()

#Finding the frauds
#There's no reverse mapping function in minisom.
#We use a dictionary from minisom.com
mappings = som.win_map(X) #returns the mappings from all the winning nodes to the customers
#the keys are the coordinates of the winning nodes; each nodes contains one or more customers
#From the colormap, we identified the nodes that appear to be outliers:
frauds = np.concatenate((mappings[(2,9)], mappings[(3,9)], mappings[(4,9)], mappings[(7,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)








