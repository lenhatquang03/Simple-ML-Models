import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir(r'C:\Users\Lenovo\Desktop\Python Bootcamp\14-K-Nearest-Neighbors')
df = pd.read_csv('KNN_Project_Data')

# Check the head of the dataframe
print(df.head())

# Use Seaborn on the dataframe to create a pairplot with hue indicated by the TARGET CLASS column
sns.pairplot(df, hue = 'TARGET CLASS', palette= 'coolwarm', diag_kind = 'hist', plot_kws = {'size': 3}, diag_kws = {'bins': 15, 'multiple': 'stack'})
plt.show()

# Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis = 1)) # Fit the scaler to the features
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1)) # Transform the features
scaled_df = pd.DataFrame(scaled_features, columns= df.columns[:-1])

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size= 0.3, random_state = 1)

# Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

# Predictions and Evaluation
from sklearn.metrics import classification_report, confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_true = y_test, y_pred = pred))
print(classification_report(y_true = y_test, y_pred = pred, target_names = ['Class 0', 'Class 1']))

# Choosing the best K value
test_error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    test_error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize = (10, 6))
plt.plot(range(1, 40), test_error_rate, marker = 'o', markerfacecolor = 'red', ls = 'dashed', color = 'blue', markersize = 10)
plt.title('Test Error Rate for each KNN classifier')
plt.xlabel('K')
plt.ylabel('Test Error Rate')
plt.show()

# With K = 32, the test error rate is the smallest. So 32 is the best value we'er looking for
best_knn = KNeighborsClassifier(n_neighbors = 39)
best_knn.fit(X_train, y_train)
best_pred = best_knn.predict(X_test)
print(classification_report(y_true = y_test, y_pred = best_pred, target_names= ['Class 0', 'Class 1']))