import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import os
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

file_path = "C:/Users/komal/Downloads/extracted_data/Car_Rates.csv"
Car_Rates = pd.read_csv(file_path)

Car_Rates.drop(['Car_name','Num_of_reviews','Year','Brand','Model'], axis=1, inplace=True)
Car_Rates.info()

Car_Rates['General_rate'].fillna(Car_Rates['General_rate'].mean(), inplace=True)

correlationMatrix = Car_Rates.corr()     
sns.heatmap(correlationMatrix, annot=True)     
plt.show()

features = Car_Rates.drop('General_rate', axis=1)
target = Car_Rates['General_rate']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=100)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linreg = linear_model.LinearRegression()
linreg.fit(X_train_scaled, y_train)

y_pred = linreg.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.figure(figsize=(8, 6))     
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)     
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  
plt.xlabel("True Values")     
plt.ylabel("Predicted Values")     
plt.title("Model Fit (True vs Predicted Values)")     
plt.show()  
 

plt.figure(figsize=(10, 6))
correlation = Car_Rates.corr()['General_rate'].sort_values(ascending=False).drop(labels=['General_rate'])
sns.barplot(y=correlation.values, x=correlation.index, palette="coolwarm")
plt.title("Correlation with General_rate")
plt.show()
