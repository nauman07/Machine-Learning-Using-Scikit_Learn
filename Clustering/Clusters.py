import sklearn.datasets
import sklearn.cluster
import sklearn.metrics
from sklearn import metrics
iris=sklearn.datasets.load_iris()
from sklearn.cluster import KMeans

kmeans_cluster = KMeans(n_clusters=3)

kmeans_cluster = kmeans_cluster.fit(iris.data) 
print(metrics.homogeneity_score(kmeans_cluster.predict(iris.data), iris.target))

from sklearn.cluster import AgglomerativeClustering
ac3 = AgglomerativeClustering(n_clusters = 3)
agg_cls3=ac3.fit_predict(iris.data)
print(metrics.homogeneity_score(agg_cls3, iris.target))

from sklearn.cluster import AffinityPropagation
af = AffinityPropagation()
af_cls=af.fit(iris.data)
print(metrics.homogeneity_score(af_cls.predict(iris.data), iris.target))
