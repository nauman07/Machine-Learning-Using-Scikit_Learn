import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

boston = datasets.load_digits()

X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, random_state=30)

print(X_train.shape)
print(X_test.shape)
from sklearn.svm import SVC
svm_classifier = SVC()
svm_classifier = svm_classifier.fit(X_train, Y_train) 
#print(svm_classifier.score(X_train,Y_train))
print(svm_classifier.score(X_test,Y_test))
#predicted = svm_classifier.predict(X_test[:2])
#print(predicted)
import sklearn.preprocessing as preprocessing

standardizer = preprocessing.StandardScaler()
standardizer = standardizer.fit(boston.data)
digit_standardized = standardizer.transform(boston.data)
X_train, X_test, Y_train, Y_test = train_test_split(digit_standardized, boston.target, random_state=30)
svm_classifier2 = SVC()

svm_classifier2 = svm_classifier2.fit(X_train, Y_train) 
print(svm_classifier2.score(X_test,Y_test))
