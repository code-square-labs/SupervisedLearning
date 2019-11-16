import matplotlib.pyplot as plt
import sklearn.datasets as data
import sklearn.linear_model as lm

# Loading boston housing data
boston = data.load_boston()

# The 6th column is the average n of room
# This is our features
X = boston.data[:, 5:6]

# Price of properties
# This is our labels
y = boston.target

# Regressor Instance
lr = lm.LinearRegression()

# Training data
lr.fit(X, y)
# Validation, 0 to 1
print(lr.score(X, y)) # Out: 0.4835254559913343

# Predict using the linear model
y_pred = lr.predict(X)

plt.scatter(X, y, marker='.', alpha=0.5, color='blue')
plt.scatter(X, y_pred, marker='.', alpha=0.5, color='red')
plt.xlim(0, 10)
plt.ylim(0, 60)
plt.xlabel('Average n of rooms')
plt.ylabel('Price ($1000)')

plt.show()
