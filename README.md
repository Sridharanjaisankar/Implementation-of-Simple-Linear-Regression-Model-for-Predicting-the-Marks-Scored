# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sridharan J 
RegisterNumber: 212222040158
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
mse=mean_squared_error(y_test,y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE =",mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='orange')
plt.plot(x_test,regressor.predict(x_test),color='red')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
```
## Output:
### df.head:
![image](https://github.com/user-attachments/assets/0878145e-eff7-41f1-9626-66728b2e193d)

### df.tail:
![image](https://github.com/user-attachments/assets/a3d5b63f-ea96-4de4-affa-3cbc78a12761)

### Array value of X:
![image](https://github.com/user-attachments/assets/77848da2-fee6-4f8b-834a-6a6bf1c318bc)

### Array value of Y:
![image](https://github.com/user-attachments/assets/fdf8eda9-ae8a-40c7-9b2a-4dbd3c531771)

### Values of Y prediction:
![image](https://github.com/user-attachments/assets/fd3d4492-100e-49f7-8dd5-def40d3c8c9e)

### Value for MSE,MAE,RMSE:
![image](https://github.com/user-attachments/assets/9f612270-ded8-4059-a9c7-b186801359e9)

### Training Set Graph:
![image](https://github.com/user-attachments/assets/a59a64f9-84fc-402e-b492-8fa24053fb08)

## Test Set Graph:
![image](https://github.com/user-attachments/assets/d5fcb706-814e-4ebd-a579-aaecbaf303e3)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
