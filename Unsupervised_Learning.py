from load_features_K import load_data , load_data_with_domain_label
import matplotlib.pyplot as plt
from Evaluation import cluster_labels
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd

Train_features, Image_labels = load_data(True)
test_features, image_labels = load_data(False)
train_features, domain_labels = load_data_with_domain_label()

model = AgglomerativeClustering(n_clusters = 5,linkage = "average")
x_pred = model.fit_predict(Train_features)



x = Train_features[:1000]
y = x_pred[:1000]

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(x)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 7),
                data=df).set(title="Iris data T-SNE projection")
import matplotlib.pyplot as plt
plt.show()