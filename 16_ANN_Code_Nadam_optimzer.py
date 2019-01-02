import theano
import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Keraslibraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('EEG_Eye_State_Arff.csv')
x = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# Initialisingthe ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim= 7,init= 'uniform', activation = 'relu', input_dim= 14))
# Adding the second hidden layer
classifier.add(Dense(output_dim= 7, init= 'uniform', activation ='relu'))
# Adding the output layer
classifier.add(Dense(output_dim= 1, init= 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'Nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size= 15, epochs= 150)

# Predicting the Test set results
y_pred= classifier.predict(x_test)
y_pred= (y_pred> 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

