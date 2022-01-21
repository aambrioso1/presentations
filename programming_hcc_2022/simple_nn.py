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
		dm += x * (y - y_hat)
		db += y - y_hat
	
	return [-2*dm/n, -2*db/n]

def optimize(X, Y, num_iterations=10, learning_rate=0.1):
	m, b = [0,0] #initialize_parameters()
	
	for i in range(num_iterations):
		dm, db = compute_gradients(X, Y, m, b)
		m = m - learning_rate * dm
		b = b - learning_rate * db
		cost = compute_cost(X, Y, m, b)
		# print(f"{cost=}, {m=},{b=},{dm=},{db=}")
	return [m, b]

num = 1000000
alpha = 0.0001
m, b = optimize(train_x, train_y, num, alpha)

print(f"m = {m} and b = {b}")

import matplotlib.pyplot as plt
plt.scatter(train_x, train_y)
x_list = []
y_list = []
for x in np.arange(0,10,0.1):
	y = m*x + b
	x_list.append(x)
	y_list.append(y)
plt.plot(x_list,y_list)
plt.show()


