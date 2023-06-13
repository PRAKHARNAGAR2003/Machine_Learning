# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set in the python program
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# print(X)
# print(Y)

# take care of the missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
# print(X)

# enter the categorical data

# encoding the independent variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

# encoding the dependent variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

# print(Y)

# Splitting the dataset into the training set and into the data set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# print(X_train)
print(X_test)