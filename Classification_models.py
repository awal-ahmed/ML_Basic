import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/drive/My Drive/dataset/processData.csv')
df.head()

number_of_features = 9
X = df.iloc[:, range(number_of_features)].values 
y = df.iloc[:, 10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=70)


# BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

clf = BaggingClassifier(base_estimator=SVC(), n_estimators=5, random_state=5)
clf.fit(X_train[:,[0,1,2,5,6]], y_train)

y_pred = clf.predict(X_test[:,[0,1,2,5,6]])


# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

clf = clf = DecisionTreeClassifier(random_state=45)
clf.fit(X_train[:,[0,1,2,5,6]], y_train)

y_pred = clf.predict(X_test[:,[0,1,2,5,6]])



# GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train[:,[0,1,2,5,6]], y_train).predict(X_test[:,[0,1,2,5,6]])


# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

regressor =  KNeighborsRegressor(n_neighbors=4)
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)


# LogisticRegression
from sklearn.linear_model import LogisticRegression

clf =  LogisticRegression(random_state=15)
clf.fit(X_train[:,[0,1,2,5,6]], y_train)

y_pred = clf.predict(X_test[:,[0,1,2,5,6]])



# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

clf =  RandomForestClassifier(max_depth=5, random_state=41)
clf.fit(X_train[:,[0,1,2,5,6]], y_train)

y_pred = clf.predict(X_test[:,[0,1,2,5,6]])

# SVC
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0 )
classifier.fit(X_train[:,[0,1,2,5,6]], y_train)

y_pred =  classifier.predict(X_test[:,[0,1,2,5,6]])


# VotingClassifier
from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[('nb', gnb), ('rf', clf1), ('lr', clf3)], voting='soft',weights=[1, 2,3])
eclf1 = eclf1.fit(X_train[:,[0,1,2,5,6]], y_train)
y_pred = eclf1.predict(X_test[:,[0,1,2,5,6]])


# ANN 
# sequential model to initialise our ann and dense module to build the layers
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9599))
regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
regressor.fit(X_train, Y_train, batch_size = 10, epochs = 100,verbose = 0)


y_pred = regressor.predict(X_test)


# Testing matrix for classification
print(y_pred.shape)
print(y_test.shape)
print('test Accuracy :',metrics.accuracy_score(y_test,y_pred))
print('Precision :',metrics.precision_score(y_test,y_pred, average='macro'))
print('Recall :',metrics.recall_score(y_test,y_pred, average='macro'))
print('F-score :',metrics.f1_score(y_test,y_pred, average='macro'))
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test,y_pred))
EPSILON = 1e-10
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Relative Absolute Error(RAE):', np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test - np.mean(y_test))) + EPSILON))
print('Root Relative Squared Error(RRSE):', np.sqrt(np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))))
