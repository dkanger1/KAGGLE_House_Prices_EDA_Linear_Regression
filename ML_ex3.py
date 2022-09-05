import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # plotting
import seaborn as sns

# leitura dos dados
housing = pd.read_csv('housing.csv')
housing.head()
#housing.isnull()
housing.dtypes

df = housing.dropna()
print(df.shape)
print(df.head(5))

df.describe()

hm = sns.heatmap(df.corr(), annot = True , cmap='coolwarm')
plt.show()

df = pd.get_dummies(df, columns = ['ocean_proximity'])
df1=df[['longitude','latitude','housing_median_age','total_rooms', 'total_bedrooms',
       'population', 'households','median_income','ocean_proximity_<1H OCEAN',
       'ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY',
       'ocean_proximity_NEAR OCEAN','median_house_value']]

X = df1.drop('median_house_value', axis=1)
y = df1['median_house_value']       

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
mlr_skl = linear_model.LinearRegression()
mlr_skl.fit(X_train, y_train.ravel())

y_pred = mlr_skl.predict(X_test) # predizando
mlr_skl.coef_  #valores finais de theta
print(f'RMSE {round(np.sqrt(mean_squared_error(y_test, y_pred )))}')

