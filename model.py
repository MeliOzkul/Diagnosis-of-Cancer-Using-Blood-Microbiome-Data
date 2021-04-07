import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xgb

data=pd.read_csv('dataset/data.csv')
label=pd.read_csv('dataset/labels.csv')

input=data.iloc[0:,1:]
target=label.iloc[0:,1]

#Colon cancer->0, lung cancer->1,breast cancer->2,prosrtate cancer->3
for i in range(0,target.shape[0]):
    print(target[i])

for i in range(0,target.shape[0]):
    if target[i]== "colon cancer":
        target[i]=0
    elif target[i]== "lung cancer":
        target[i]=1
    elif target[i] == "breast cancer":
        target[i] = 2
    elif target[i]== "prosrtate cancer":
        target[i]=3

print(input.shape)
print(target.shape)

X=np.array(input).astype('float64')
y=np.array(target).astype('uint8')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Mean Absolute Error For Random Forest:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error For Random Forest:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error For Random Forest:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Score For Random Forest:',regressor.score(X_test,y_test))

XGBregressor = xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)

XGBregressor.fit(X_train,y_train)
xgb_pred=XGBregressor.predict(X_test)

print('Mean Absolute Error For XGBoost:', metrics.mean_absolute_error(y_test, xgb_pred))
print('Mean Squared Error For XGBoost:', metrics.mean_squared_error(y_test, xgb_pred))
print('Root Mean Squared Error For XGBoost:', np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))
print('Score For XGBoost:',regressor.score(X_test,y_test))