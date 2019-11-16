# Supervised Learning
It is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.

I am using *boston housing* data sets, which is provided by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)

## Linear Regression
The linear regression algorithm is one of the fundamental supervised machine-learning algorithms

Formula: `y = a*x + b`.

`y` is the dependent variable (that's the variable that goes on the y axis)

`x` is the independent variable (i.e. it is plotted on the x axis)

`b` is the slope of the line

`a` is the y-intercept

Out put of [boston-housing.py](https://github.com/mertaksoy/BostonHousing/blob/master/boston-housing.py):

Manually Loss-Function (a = 15.; b = -70.)
![Linear Regression](https://raw.githubusercontent.com/mertaksoy/BostonHousing/master/boston-housing.png "Linear Regression")

### Ordinary Least Squares
Lets use OLS for estimating the unknown parameters `a` and `b`

Out put of [ordinary-least-square.py](https://github.com/mertaksoy/BostonHousing/blob/master/ordinary-least-square.py):

Estimated Loss Function
![Estimated Loss Function](https://raw.githubusercontent.com/mertaksoy/BostonHousing/master/ordinary-least-square.png "Estimated Loss Function")

### Stochastic Gradient Descent
Another algorithm for estimating the unknown parameters `a` and `b`

Out put of [stochastic-gradient-descent.py](https://github.com/mertaksoy/BostonHousing/blob/master/stochastic-gradient-descent.py):

Estimated Loss Function via Stochastic Gradient Descent
![Estimated Loss Function](https://raw.githubusercontent.com/mertaksoy/BostonHousing/master/stochastic-gradient-descent.png "Estimated Loss Function")

