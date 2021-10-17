import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
#np.random.seed(100)
max_depth = range(2, 6)

boston = datasets.load_boston()

X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, random_state=30)

print(X_train.shape)
print(X_test.shape)

from sklearn.ensemble import RandomForestRegressor
rf_classifier = RandomForestRegressor()
rf_classifier = rf_classifier.fit(X_train, Y_train)
print(round(rf_classifier.score(X_train,Y_train),1))

print(round(rf_classifier.score(X_test,Y_test),1))
predicted = rf_classifier.predict(X_test[:2])
print(predicted)
a=0
b=0
score=0
it=[50,100,200]
for i in range(3,6):
  for j in it:
    rf_classifier = RandomForestRegressor(n_estimators=j,max_depth=i)
    rf_classifier = rf_classifier.fit(X_train, Y_train) 
    s=rf_classifier.score(X_test,Y_test)
    if s>score:
      score=s
      a=i
      b=j
    else:
      pass
tup=(a,b)
tup=tuple(tup)
print(tup)

