---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: Python [conda env:root] *
    language: python
    name: conda-root-py
---

```python
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
warnings.filterwarnings('ignore')
```

# Question 0

```python
def split_x_y(data,label):
    x=data.drop(label,axis=1)
    y=data[label]
    return x.values,y.values
```

```python
data=pd.read_csv("./train.csv")
cols=data.columns
# y=data['critical_temp']
n_of_ele=data['number_of_elements']
train,test_valid=train_test_split(data,test_size=0.2,shuffle=True,stratify=n_of_ele)
n_of_ele=test_valid['number_of_elements']
test,valid=train_test_split(test_valid,test_size=0.5,shuffle=True,stratify=n_of_ele)
```

# Question 1


## part a

```python
# if set normalize=False,the objective dont converge
model=ElasticNet(normalize=True)
train_x,train_y=split_x_y(train,'critical_temp')
param=[
    {'l1_ratio':list(np.arange(0.,1,0.1))}
]
grida=GridSearchCV(ElasticNet(),param,cv=10,scoring='neg_mean_squared_error')
grida.fit(train_x,train_y)
```

```python
param=grida.cv_results_['params']
score=grida.cv_results_['mean_test_score']
param=[i['l1_ratio']for i in param]
plt.plot(param,score)
```

```python
grida.best_estimator_
```

## part b

```python
model=RandomForestRegressor()
param=[
    {'n_estimators':list(range(1,10,2)),'max_depth':list(range(30,40,3))}
]
gridb=GridSearchCV(model,param,cv=10,scoring='neg_mean_squared_error')
gridb.fit(train_x,train_y)
```

```python
gridb.cv_results_
```

```python
param=gridb.cv_results_['params']
score=gridb.cv_results_['mean_test_score']

n_est=[i['n_estimators']for i in param]
max_depth=[i['max_depth']for i in param]

```

```python
plt.scatter(n_est,max_depth,c=score)
plt.xlabel("num of trees")
plt.ylabel("max depth of trees")
plt.colorbar()
```

```python
#from the figure we can inference that the more trees, the better the model performance.
gridb.best_estimator_
#And the best param is (max_depth=33, n_estimators=9)
```

## part c

```python
model=GradientBoostingRegressor(learning_rate=0.05)
param=[
    {'n_estimators':list(range(1,10,2))}
]
gridc=GridSearchCV(model,param,cv=10,scoring='neg_mean_squared_error')
gridc.fit(train_x,train_y)
```

```python
gridc.cv_results_
```

```python
params=gridc.cv_results_['params']
n_est=[i['n_estimators'] for i in params]
score=gridc.cv_results_["mean_test_score"]
```

```python
plt.plot(n_est,score)
```

```python
gridc.best_estimator_
```

# Question 2

```python
grid_res=[grida,gridb,gridc]
grida.best_params_
```

```python
grid_res=[grida,gridb,gridc]
best_params=[gr.best_params_ for gr in grid_res]
best_params
```

```python
model1=ElasticNet(**best_params[0])
model2=RandomForestRegressor(**best_params[1])
model3=GradientBoostingRegressor(**best_params[2])
best_model=[model1,model2,model3]
for i in best_model:
    i.fit(train_x,train_y)
y_pre=[]
valid_x,valid_y=split_x_y(valid,'critical_temp')
for model in best_model:
    y_pre.append(model.predict(valid_x))
```

```python
loss=[]
for y in y_pre:
    loss.append(mean_squared_error(y,valid_y))
```

```python
table=pd.DataFrame(columns=['ElasticNet','RandomForestRegressor','GradientBoostingRegressor'])
table.loc[0]=loss
table
```

```python
# best model is RandomForestRegressor
best=best_model[1]
test_x,test_y=split_x_y(test,"critical_temp")
res=best.predict(test_x)
```

```python
res
```

```python
# the corresponding MSE is 91.28
mean_squared_error(res,test_y)
```

```python

```
