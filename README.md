# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph.
 
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sridharan J 
RegisterNumber: 212222040158
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output:
![alt text](<Screenshot 2025-03-08 142541.png>)

## Program:
```
dataset.info()
```
## Output:
![alt text](<Screenshot 2025-03-08 142549.png>)

## Program:
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
## Output:
![alt text](<Screenshot 2025-03-08 142602.png>)

## Program:
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
```
## Output:
![alt text](<Screenshot 2025-03-08 142612.png>)

## Program:
```
X_test.shape
```
## Output:
![alt text](<Screenshot 2025-03-08 142618.png>)

## Program:
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
## Output
![alt text](<Screenshot 2025-03-08 142626.png>)

## Program:
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
## Output:
![alt text](<Screenshot 2025-03-08 142634.png>)

## Program:
```
plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='yellow')
plt.title('Training Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
## Output:
![alt text](<Screenshot 2025-03-08 142647.png>)

## Program:
```
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='silver')
plt.title('Test Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
## Output:
![alt text](<Screenshot 2025-03-08 142711.png>)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
