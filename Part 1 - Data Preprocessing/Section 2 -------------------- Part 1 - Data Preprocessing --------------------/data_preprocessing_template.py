# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #first colon means we take all the lines and second colon after , means take all columns except last one - thats is denoted by -1 
y = dataset.iloc[:, 3].values


# This is how we are going to fill missing data, by filling with mean of the columns data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy="mean",axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Now we need to encode columns which contains categories
# like country as it contains three categories France, Germany and Spain as 0,1,2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])

#But for y(purchased) which is a dependent variable and also contains categories yes or no, we do only Label Encoding
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Now we need to Dummy Encoding, as we need to avoid machine learning thinking,
# that Germany is greater than France or Spain is greater than Germany
oneHotEncode = OneHotEncoder(categorical_features=[0])
X = oneHotEncode.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)