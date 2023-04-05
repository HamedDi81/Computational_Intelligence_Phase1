from load_features_H import load_data , load_data_with_domain_label
from Evaluation import apply
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

Train_features, image_labels = load_data(True)
print(Train_features.shape)
print(image_labels.shape)
print(f'Number of training samples: {Train_features.shape[0]}')
test_features, image_labels = load_data(False)
print(test_features.shape)
print(image_labels.shape)
train_features, domain_labels = load_data_with_domain_label()
print(train_features.shape)
print(domain_labels.shape)
kmeans = KMeans(n_clusters = 10)
model=kmeans.fit(Train_features)
x_pred = model.predict(Train_features)
# test_pred = model.predict(x_test)
print(f"shape of pred x is {x_pred.shape}")
# apply(Train_features,image_labels)


