import matplotlib.pyplot as plt
import sklearn.datasets as data
import numpy as np

boston = data.load_boston()

# The 6th column is the average n of room
# This is our features
X = boston.data[:, 5:6]

# Price of properties
# This is our labels
y = boston.target


'''
Linear Regression
y = a*x + b

y -> is the dependent variable (price)
x -> is the independent variable (rooms)
b -> the slope of the line and a is the y-intercept
a -> the y-intercept
'''
a = 15.; b = -70.
lx = np.arange(4,10) # Returns evenly spaced values within 4 and 10
lguess = a*lx + b

plt.plot(lx, lguess, c='red')
plt.scatter(X, y, marker='.', alpha=0.5, color='blue')
plt.xlim(0, 10) 
plt.ylim(0, 60)
plt.xlabel('Average n of rooms')
plt.ylabel('Price ($1000)')

plt.show()
