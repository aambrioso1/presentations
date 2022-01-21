# simple_nn.py

# A simple neural network with one node that has two inputs and one output.

import math
import random as rd
import numpy as np

train_x = [1, 2, 3, 5, 8]
train_y = [1, 3, 2, 6, 12]

def initialize_parameters():
	m = rd.random()
	b = rd.random()
	return [m, b]

def compute_cost(X,Y,m,b):
	n = len(X)
	J = 0
	for i, x in enumerate(X):
		y_hat = m * x + b
		y = Y[i]
		J += (y - y_hat) ** 2
	return J/n

def compute_gradients(X, Y, m, b):
	n = float(len(X))
	dm = 0
	db = 0

	for i, x in enumerate(X):
		y = Y[i]
		y_hat = m*x + b
		dm += x * (y - y_hat) # partial with respect to m of cost function
		db += y - y_hat # partial with respect to b of cost function
	
	return [-2*dm/n, -2*db/n]

def optimize(X, Y, num_iterations=10, learning_rate=0.1):
	m, b = [0,0] # initialize parameters
	# m, b = initialize_parameters() # Use for random initialization
	line_list = []
	
	for i in range(1, num_iterations+1):
		# gradients are used to update the parameters (backward propagation)
		dm, db = compute_gradients(X, Y, m, b)
		m = m - learning_rate * dm
		b = b - learning_rate * db
		graph_count = 5

		# Pick 5 lines to graph as algorithm learns regression line
		# Could be improved by pick lines at an exponential rate
		if i in [1, 10, 100, 1000, 10000, 100000, 100000]:
		# if i % (num_iterations//graph_count) == 0:
			line_list.append([m,b])
			print(f"iterations = {i}")
		# recompute the cost (forward propagation)
		cost = compute_cost(X, Y, m, b)
		# print(f"{cost=}, {m=},{b=},{dm=},{db=}")
	return line_list

num = 1000000
alpha = 0.0001
learned_line_list = optimize(train_x, train_y, num, alpha)

print(f"The learned line list is {learned_line_list}")

import matplotlib.pyplot as plt
plt.scatter(train_x, train_y, color="red")
x_list = []
y_list = []
for i, line in enumerate(learned_line_list):
	m, b = line
	for x in np.arange(0,10,0.1):
		y = m*x + b
		x_list.append(x)
		y_list.append(y)
		plt.plot(x_list,y_list, linewidth=1.0, color="blue")
plt.show()


