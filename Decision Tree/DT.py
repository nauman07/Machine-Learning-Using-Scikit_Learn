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

dt_reg = DecisionTreeRegressor()
dt_reg = dt_reg.fit(X_train, Y_train)

print(round(dt_reg.score(X_train,Y_train),1))
print(round(dt_reg.score(X_test,Y_test),1))
predicted = dt_reg.predict(X_test[:2])
print(predicted)
dt_reg = DecisionTreeRegressor()
#random_grid = {'max_depth': max_depth}
#dt_random = RandomizedSearchCV(estimator = dt_reg, param_distributions = random_grid, 
#n_iter = 90, cv = 3, verbose=2, random_state=42, n_jobs = -1)
a=0
b=0
for i in range(2,6):
    dt_reg = DecisionTreeRegressor(max_depth=i)
    dt_reg = dt_reg.fit(X_train, Y_train)
    score=dt_reg.score(X_test,Y_test)
    if score>b:
        b=score
        a=i
    else:
        pass
print(a)

