# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets
2. Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters
3. Train your model -Fit model to training data -Calculate mean salary value for each subset
4. Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
5. Tune hyperparameters -Experiment with different hyperparameters to improve performance
6. Deploy your model Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SUBASH R
RegisterNumber: 212223230218
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:
![image](https://github.com/user-attachments/assets/4d3c9634-a569-4585-9ffd-538e343c39f7)
![image](https://github.com/user-attachments/assets/11fe9207-7427-47d3-97bc-98ba42f7188d)
![image](https://github.com/user-attachments/assets/7b59c261-b32d-4daf-b8bc-4f5485f3282e)
![image](https://github.com/user-attachments/assets/42edbaff-19af-4847-b383-61c8ae96df48)
![image](https://github.com/user-attachments/assets/b5c26350-4d83-46dc-8051-76971c884e6f)
![image](https://github.com/user-attachments/assets/0dbc1026-079b-4cf5-b518-f397aba80317)
![image](https://github.com/user-attachments/assets/f75c0090-7c13-4f29-8821-fb72b6cb8cdf)
![image](https://github.com/user-attachments/assets/263e3c57-9fd3-4967-9e7b-bbf7a485b8b3)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
