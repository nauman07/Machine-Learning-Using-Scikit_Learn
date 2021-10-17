import sklearn.datasets
import sklearn.model_selection
from sklearn.model_selection import train_test_split
iris=sklearn.datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target,stratify=iris.target,random_state=30)
print(X_train.shape)
print(X_test.shape)
#task
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()   
knn_classifier = knn_classifier.fit(X_train, Y_train) 
print(round(knn_classifier.score(X_train,Y_train),1))
print(round(knn_classifier.score(X_test,Y_test),1))
score=0
a=0
for i in range(3,11):
  knn_classifier = KNeighborsClassifier(n_neighbors=i)   
  knn_classifier = knn_classifier.fit(X_train, Y_train)
  s=knn_classifier.score(X_test,Y_test)
  if s>score:
    score=s
    a=i
print(a)


