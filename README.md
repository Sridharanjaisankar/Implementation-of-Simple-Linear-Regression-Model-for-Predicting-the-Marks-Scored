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
![Screenshot 2025-04-07 103056](https://github.com/user-attachments/assets/cd612d53-c0b1-47e3-86b5-cbefce4e3c05)

## Program:
```
dataset.info()
```
## Output:
![image](https://github.com/user-attachments/assets/cec9458e-2aa6-4a67-9f30-7ee337643ee6)

## Program:
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
## Output:
![image](https://github.com/user-attachments/assets/a721fd7c-946d-4bae-80dc-ad440f4d6dca)

## Program:
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
```
## Output:
![image](https://github.com/user-attachments/assets/85d54c93-a000-4b57-8528-b90ff06f4411)

## Program:
```
X_test.shape
```
## Output:
![image](https://github.com/user-attachments/assets/22e896cd-722f-4dbf-82bd-2ca2b24dda89)

## Program:
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
## Output
![image](https://github.com/user-attachments/assets/8ffabef6-2e47-4d3b-a580-fa3f6f074850)

## Program:
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
## Output:
![image](https://github.com/user-attachments/assets/bd9f3d3e-a3b9-408b-a495-d8cea3bc75c3)

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
![image](https://github.com/user-attachments/assets/9e35b0de-6295-4b03-b022-f5233e0d759f)

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
![image](https://github.com/user-attachments/assets/34d8fc8f-6f4c-4ae2-a972-0234c39bb97a)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
