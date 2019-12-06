import calendar
import pandas as pd
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from math import sqrt

#read data sets
df = pd.read_csv("train_cab.csv")
df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC',errors='coerce' )
df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors='coerce')

#clean date_time feature
df = df[df["pickup_datetime"] < df["pickup_datetime"].max()]
df = df[df["pickup_datetime"] > df["pickup_datetime"].min()]
df.reset_index(inplace=True)
df = df.drop(['index'], axis= 1)

#creation of hour,day,day_of_week,month,year
df['pickup_day'] = df['pickup_datetime'].apply(lambda x: x.day)
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
df['pickup_day_of_week'] = df['pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: x.month)
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: x.year)

#calculating distance
def distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
X = []
for i in range(len(df)):
    lat1 = df["pickup_latitude"][i]
    lat2 =df["dropoff_latitude"][i]
    lon1 =df["pickup_longitude"][i]
    lon2 =df["dropoff_longitude"][i]
    d = distance(lat1,lat2,lon1,lon2)
    X.append(d)
print(X[0:10])

df["Distance"] = X

#removing outliers
len(df.loc[df["passenger_count"] > 8])
df = df[df["Distance"] < 2000]
df = df[df["Distance"] > 0 ]
df = df[df["passenger_count"] >= 1]
df = df[df["passenger_count"] <= 8]
df = df.loc[df["fare_amount"] >= 1]
df = df.loc[df["fare_amount"] <4000 ]

print(df.describe())

#convery pick up day of week into numeric
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['pickup_day_of_week'].drop_duplicates())
df['pickup_day_of_week'] = encoder.transform(df['pickup_day_of_week'])

#correlation for feature selection
df_corr = df.drop(["pickup_datetime"], axis= 1)
corr = df_corr.corr()
#p-value calculation
import statsmodels.formula.api as sm
x = (df_corr.iloc[:,5:])
Y = (df_corr.iloc[:, 0])
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues)

        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())

    return x
SL = 0.05
data_modeled = backwardElimination(x.values, SL )

#split train data into train and test
from sklearn.model_selection import train_test_split
X = df_corr.iloc[:,-3:]
y = df_corr[["fare_amount"]]
df_train, df_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df_train, y_train)
#X_test = df_test[[""]]
y_pred = reg.predict(df_test)

#visualising train set
plt.scatter(df_train[['Distance']],y_train,color='red')
plt.plot(df_train[['Distance']],reg.predict(df_train),color='blue')
plt.title('Distance vs fare_amount(train_set)')
plt.xlabel('distance')
plt.ylabel('fare_amount')

#visualising test set
plt.scatter(df_test[['Distance']],y_test,color='red')
plt.plot(df_train[['Distance']],reg.predict(df_train),color='blue')
plt.title('Distance vs fare_amount(test_set)')
plt.xlabel('distance')
plt.ylabel('fare_amount')

#error rate
from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, y_pred)
print(score)
root_mean= sqrt(score)
print("RMSE of linear regression:", root_mean)

#decision tree
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(max_depth=500)
dt.fit(df_train,y_train)
dt_pred=dt.predict(df_test)

#error rate for dt
from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, dt_pred)
print(score)
root_mean= sqrt(score)
print("RMSE of decision tree:", root_mean)

#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=200)
rf.fit(df_train, y_train)
y_dec_pred = rf.predict(df_test)

#error rate for rf
from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test, y_dec_pred)
root_mean= sqrt(score)
print("RMSE for Random forest:", root_mean)

#read test data
test = pd.read_csv("test.csv")

#date time format
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC',errors='coerce' )

#creation of variables from date time feature
test['pickup_day'] = test['pickup_datetime'].apply(lambda x: x.day)
test['pickup_hour'] = test['pickup_datetime'].apply(lambda x: x.hour)
test['pickup_day_of_week'] = test['pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
test['pickup_month'] = test['pickup_datetime'].apply(lambda x: x.month)
test['pickup_year'] = test['pickup_datetime'].apply(lambda x: x.year)

#distance for test
X_testfare = []
for i in range(len(test)):
    lat1 = test["pickup_latitude"][i]
    lat2 =test["dropoff_latitude"][i]
    lon1 =test["pickup_longitude"][i]
    lon2 =test["dropoff_longitude"][i]
    d = distance(lat1,lat2,lon1,lon2)
    X_testfare.append(d)
print(X_testfare[0:10])

test["Distance"] = X_testfare

#imputing random forest model into test data as it has lowest RMSE value
test_to=test[['pickup_month','pickup_year','Distance']]
test_model=rf.predict(test_to)
test["predicted_fareamount"]=test_model


