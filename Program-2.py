# regression 

# Simple linear regression

# importing the libraries:-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset: - 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# splitting the Dataset into the training set and the test set: -
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X , Y , test_size=0.2, random_state=0)

# Training the simple regression model for on the Training set:- 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test set result:-
Y_pred = regressor.predict(X_test) # here Y_pred contain all the predicted salaries and Y_test cntsin all the real salaries.

# Visualizing the training set results:-
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the test set results: - 
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()