import matplotlib.pyplot as plt
import sklearn.datasets as data
import sklearn.linear_model as lm

def determine(iterations, lineColor):
	lr = lm.SGDRegressor(n_iter_no_change=iterations)
	lr.fit(X, y)
	print(lr.score(X, y))
	y_pred = lr.predict(X)
	plt.scatter(X, y_pred, marker='.', alpha=0.5, color=lineColor)

# Loading boston housing data
boston = data.load_boston()

# The 6th column is the average n of room
# Those are our features
X = boston.data[:, 5:6]

# Price of properties
# Those are our labels
y = boston.target

# The value of n_iter_no_change is default 5
# Score: 0.3457659979699824
determine(5, 'red')

# Lets try with more iterations and compare the result
# Score: 0.47496216327736646
determine(10000, 'black')

plt.scatter(X, y, marker='.', alpha=0.5, color='blue')
plt.xlim(0, 10)
plt.ylim(0, 60)
plt.xlabel('Average n of rooms')
plt.ylabel('Price ($1000)')

plt.show()
