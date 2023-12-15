

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data collection and analysis"""

diabetes_data = pd.read_csv('D:\\CodeStuff\\Coding\\ML using python\\Diabetesprediction(sklearn)\\diabetes.csv')

diabetes_data.head()

diabetes_data.tail()

#no of rows and columns
diabetes_data.shape

diabetes_data['Outcome'].value_counts()

"""0 -- Not diabetic,
1 -- diabetic
"""

diabetes_data.describe()

diabetes_data.groupby('Outcome').mean()

#seprating the data and the labels

x = diabetes_data.drop(columns = 'Outcome', axis = 1)
y = diabetes_data['Outcome']

print(x)

print(y)

"""Data standerdization"""

scaler = StandardScaler()

standardized_data = scaler.fit_transform(x)

print(standardized_data)

x = standardized_data
y = diabetes_data['Outcome']

#splitting data into traning and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, stratify = y, random_state = 2)

print(x_train.shape,x_test.shape,x.shape)

"""Traning THe model"""

classifier = svm.SVC(kernel = 'linear')

#traning the support vector machine
classifier.fit(x_train,y_train)

#Model Evaluation
#Accuracy score in the traning data

x_train_prediction = classifier.predict(x_train)
traning_data_accuracy = accuracy_score(x_train_prediction,y_train)

print('Accuracy score of the traning data : ',traning_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)

print('Accuracy score of the traning data : ',test_data_accuracy)









"""Making a Predictive system"""

input_data = (13,145,82,19,110,22.2,0.245,57)

#changing the input_data to numpy array
numpy_array_data = np.asarray(input_data)

# reshape the array as we are predicting only for one instance
input_data_reshaped = numpy_array_data.reshape(1,-1)

#standerdizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)

if prediction[0]  == 0:
  print("Not Diabetic")
elif prediction[0] == 1:
  print("Diabetic")

