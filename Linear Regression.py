# The company is trying to decide whether to focus their efforts on their mobile app experience or on their website.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.chdir(r'C:\Users\Lenovo\Desktop\Python Bootcamp\11-Linear-Regression')

data = pd.read_csv('Ecommerce Customers')
print(data.head())
print(data.info())
print(data.describe())

# Exploratory Data Analysis
  # Time on Website vs Yearly Amount Spent
sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = data, joint_kws= {'color: red'})
  # Time on App vs Yearly Amount Spent
sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', data = data, joint_kws= {'color': 'red'})
  # Time on App vs Length of Membership
sns.jointplot(x = 'Time on App', y = 'Length of Membership', data = data, kind = 'hex') 

print(data.corr(method = 'pearson'))
# The correlation between Time on Website and Yearly Amount Spent (0.499) suggests as the Time on App increase, so does the Yearly Amount Spent.
# The correlation between Time on App and Yearly Amount Spent (-0.0026) doesn't make sense.
# The correlation between Time on App and Length of Membership (0.029) doesn't indicate much.

# Relationship for the entire data set
g = sns.pairplot(data = data, diag_kind = 'hist')
g.map_upper(sns.scatterplot)
g.map_lower(sns.scatterplot)
# Based off this plot, the most correlated feature with Yearly Amount Spent is the Length of Membership

# Linear model plot between Yearly Amount Spent and the Length of Membership
sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = data, scatter_kws = {'color': 'red', 'edgecolors': 'black'})

# Training and testing data
X, y = data[data.columns[3:-1]], data['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
   
# Training the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)

# Predicting the Test Data
predictions = model.predict(X_test)
plt.scatter(y = predictions, x = y_test, color = 'red', edgecolors = 'black')
plt.xlabel('True values of Y')
plt.ylabel('Predicted values of Y')

# Evaluating the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
print('MSE:', mean_squared_error(y_true = y_test, y_pred = predictions))
print('MAE:', mean_absolute_error(y_true = y_test, y_pred = predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_true = y_test, y_pred = predictions)))

# Residuals: Residual plots are useful graphical tools for identifying non-linearity. 
residuals = y_test - predictions
checked_df = pd.DataFrame()
checked_df['Residuals'] = list(residuals)
checked_df['Predicted values'] = predictions
sns.lmplot(x = 'Predicted values', y = 'Residuals', data = checked_df, scatter_kws = {'color': 'red', 'edgecolors':'black'})
plt.show()

print(checked_df.corr(method = 'pearson'))
# With a very small correlation between the residuals and predicted values, along with no discernible patterns detected from the residual plot, we have obtain a quite good model.

# Conclusions and interpretations from the coefficients
coefficients = pd.DataFrame(data = model.coef_, index = X.columns, columns = ['Coefficients'])
print(coefficients)
# Holding all other predictors fixed, a one-unit increase in Time on App is associated with an increase of approximatly 26 units in the Yearly Amount Spent.
# Holding all other features fixed, a one-unit increase in Time on Website is associated with an increase of 0.19 in the Yearly Amount Spent.
# As a result, they should focus their efforts on THEIR MOBILE APP.