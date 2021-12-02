from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
def split_x_y(data,label):
    x=data.drop(label,axis=1)
    y=data[label]
    return x.values,y.values
data=pd.read_csv("./train.csv")
cols=data.columns
# y=data['critical_temp']
n_of_ele=data['number_of_elements']
train,test_valid=train_test_split(data,test_size=0.2,shuffle=True,stratify=n_of_ele)
n_of_ele=test_valid['number_of_elements']
test,valid=train_test_split(test_valid,test_size=0.5,shuffle=True,stratify=n_of_ele)
train_x,train_y=split_x_y(train,'critical_temp')
valid_x,valid_y=split_x_y(valid,'critical_temp')
test_x,test_y=split_x_y(valid,'critical_temp')
model=RandomForestRegressor(max_depth=30, n_estimators=9)
cv=cross_val_score(model,train_x,train_y,scoring="neg_mean_squared_error",cv=10,n_jobs=10)
print(f"the mean squared error of model is {-cv.mean()}(10 fold validation)")