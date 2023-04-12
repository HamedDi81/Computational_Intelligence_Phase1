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

model = AgglomerativeClustering(n_clusters=None,linkage = "single", distance_threshold=0.5)
x_pred = model.fit_predict(Train_features)


from sklearn.decomposition import PCA
from numpy import reshape
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(123)
plt.figure(figsize=[20,20])
tsne = PCA(n_components=2)
j = 1
for i in range(10000,19000,1000):
    x = Train_features[i:i+1000]
    y = model.labels_[i:i+1000]
    z = tsne.fit_transform(x)
    
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    plt.subplot(5,2,j)
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", np.unique(y).shape[0]),
                    data=df).set(title=f"PCA data projection from {i} to {i+1000}th data")
    j+=1
plt.show()