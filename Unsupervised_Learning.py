from load_features_H import load_data , load_data_with_domain_label
import matplotlib.pyplot as plt
from Evaluation import cluster_labels
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering , MeanShift , Birch , OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


Train_features, Image_labels = load_data(True)
test_features, image_labels = load_data(False)
train_features, domain_labels = load_data_with_domain_label()


# model = AgglomerativeClustering(n_clusters=10,linkage= "ward")
model = KMeans(n_clusters=8)
x_pred = model.fit_predict(Train_features)


from sklearn.decomposition import PCA
from numpy import reshape
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
print("done")
print(np.unique(model.labels_).shape)
np.random.seed(123)
plt.figure(figsize=[40,40])
tsne =TSNE(n_components=2)
x = Train_features
plt.figure(figsize=[25,25])
plt.subplot(2,1,1)
x_pred = model.predict(train_features)
x_pred[x_pred == 1] = 3
x_pred[x_pred == 4] = 6
x_pred[x_pred == 5] = 2
z = tsne.fit_transform(train_features)
df = pd.DataFrame()
df["y"] = x_pred
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", np.unique(y).shape[0]),
                data=df).set(title=f"PCA data projection from  to th data")
plt.subplot(2,1,2)
df = pd.DataFrame()
df["y"] = domain_labels
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", np.unique(y).shape[0]),
                data=df).set(title=f"PCA data projection from  to th data")
plt.show()
plt.show()