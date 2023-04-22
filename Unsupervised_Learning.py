from load_features_K import load_data , load_data_with_domain_label
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score,silhouette_score,calinski_harabasz_score
from Evaluation import cluster_labels
from sklearn import datasets
from sklearn.decomposition import PCA,FastICA
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from numpy import reshape
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

st = StandardScaler()

Train_features, Image_labels = load_data(True)
test_features, iMage_labels = load_data(False)
train_features, image_labels = load_data_with_domain_label()
Train_features , test_features , train_features = st.fit_transform(Train_features) , st.fit_transform(test_features) , st.fit_transform(train_features)


model = AgglomerativeClustering(n_clusters=10 , linkage='ward')
x_pred = model.fit_predict(Train_features)
print(davies_bouldin_score(Train_features , model.labels_),silhouette_score(Train_features , model.labels_),calinski_harabasz_score(Train_features , model.labels_))
model = KMeans(n_clusters=10)
x_pred = model.fit_predict(Train_features)
print(davies_bouldin_score(Train_features , model.labels_),silhouette_score(Train_features , model.labels_),calinski_harabasz_score(Train_features , model.labels_))

#model improvement by reducing K down to 8

model = KMeans(n_clusters=8,verbose=1,max_iter=900)
model.fit(Train_features)
x_pred = model.predict(Train_features)

print(davies_bouldin_score(Train_features , model.labels_),silhouette_score(Train_features , model.labels_),calinski_harabasz_score(Train_features , model.labels_))

#more improvement (K = 5)

model = KMeans(n_clusters=5,verbose=1,max_iter=900)
model.fit(Train_features)
x_pred = model.predict(Train_features)

print(davies_bouldin_score(Train_features , model.labels_),silhouette_score(Train_features , model.labels_),calinski_harabasz_score(Train_features , model.labels_))

#Train data visualization

np.random.seed(123)

Pca =PCA(n_components=2)

x = Train_features

plt.figure(figsize=[25,25])

z = Pca.fit_transform(x)

df = pd.DataFrame()

df["y"] = x_pred
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", np.unique(x_pred).shape[0]),
                data=df).set(title=f"PCA data projection from  to th data")
# Domain test data visualization

x = train_features

plt.figure(figsize=[25,25])

x_pred = model.predict(train_features)

z = Pca.fit_transform(x)

df = pd.DataFrame()

df["y"] = x_pred

df["comp-1"] = z[:, 0]

df["comp-2"] = z[:, 1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", np.unique(x_pred).shape[0]),
                data=df).set(title=f"PCA data projection from  to th data")

#Clustering evaluation

cluster_labels = []
for i in np.unique(x_pred):

    c = (x_pred == i)
    label, count = np.unique(image_labels[c], return_counts=True)
    cluster_labels.append(label[np.argmax(count)])

print(cluster_labels)               #print the assigned values
x2 = x_pred.copy()                    #store a copy of predicted test datas
y_pred = np.select([x_pred == i for i in np.unique(x_pred)],cluster_labels,x_pred) #replace each value of predicted test datas
print(f"Accuracy: {accuracy_score(image_labels,y_pred) * 100}%")

# Image label evaluation

scores = np.empty([5,])
x_pred = model.predict(test_features)

for i in range(5):
    scores[i] = accuracy_score(iMage_labels[x_pred == i] , x_pred[x_pred == i])
print(f"Image Label Accuracy: {scores.mean() * 100}%")

plt.show()