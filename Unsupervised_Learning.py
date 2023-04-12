from load_features_H import load_data , load_data_with_domain_label
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
from mpl_toolkits.mplot3d import Axes3D


Train_features, Image_labels = load_data(True)
test_features, image_labels = load_data(False)
train_features, domain_labels = load_data_with_domain_label()

model = AgglomerativeClustering(n_clusters=None,linkage = "average", distance_threshold=40.45)
x_pred = model.fit_predict(Train_features)


from sklearn.decomposition import PCA
from numpy import reshape
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
print("done")
np.random.seed(123)
plt.figure(figsize=[40,40])
tsne = PCA(n_components=2)
x = Train_features
y = model.labels_
z = tsne.fit_transform(x)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", np.unique(y).shape[0]),
                data=df).set(title=f"PCA data projection from {i} to {i+1000}th data")
plt.show()