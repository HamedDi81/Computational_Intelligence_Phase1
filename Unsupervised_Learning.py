from load_features_H import load_data , load_data_with_domain_label
from Evaluation import cluster_labels
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.model_selection import train_test_split

Train_features, Image_labels = load_data(True)
print(Train_features.shape)
print(Image_labels.shape)
print(f'Number of training samples: {Train_features.shape[0]}')
test_features, image_labels = load_data(False)
print(test_features.shape)
print(image_labels.shape)
train_features, domain_labels = load_data_with_domain_label()
print(train_features.shape)
print(domain_labels.shape)
db = DBSCAN(eps = 20.2 , min_samples=10)
kmeans = KMeans(n_clusters= 5 )

# # dbscan = GaussianMixture()
# # model=kmeans.fit(Train_features)
model=kmeans.fit(train_features)
# x_pred = db.fit_predict(train_features)
x_pred= model.predict(train_features)
print(f"shape of pred x is {x_pred.shape}")
cm = confusion_matrix(domain_labels,x_pred)
ari = adjusted_rand_score(domain_labels , x_pred)
# accuracy_score
# count =np.zeros([5,5])
# for i,data in enumerate(x_pred):
#     if data==0:
#         if domain_labels[i] ==0:
#             count[0,0] +=1
#         elif domain_labels[i] == 1:
#             count[0,1] +=1
#         elif domain_labels[i]==2 :
#             count[0,2] +=1
#         elif domain_labels[i] == 3:
#             count[0,3] +=1
#         elif domain_labels[i]==4 :
#             count[0,4] +=1
#     if data==1:
#         if domain_labels[i] ==0:
#             count[1,0] +=1
#         elif domain_labels[i] == 1:
#             count[1,1] +=1
#         elif domain_labels[i]==2 :
#             count[1,2] +=1
#         elif domain_labels[i] == 3:
#             count[1,3] +=1
#         elif domain_labels[i]==4 :
#             count[1,4] +=1
#     if data==2:
#         if domain_labels[i] ==0:
#             count[2,0] +=1
#         elif domain_labels[i] == 1:
#             count[2,1] +=1
#         elif domain_labels[i]==2 :
#             count[2,2] +=1
#         elif domain_labels[i] == 3:
#             count[2,3] +=1
#         elif domain_labels[i]==4 :
#             count[2,4] +=1
#     if data==3:
#         if domain_labels[i] ==0:
#             count[3,0] +=1
#         elif domain_labels[i] == 1:
#             count[3,1] +=1
#         elif domain_labels[i]==2 :
#             count[3,2] +=1
#         elif domain_labels[i] == 3:
#             count[3,3] +=1
#         elif domain_labels[i]==4 :
#             count[3,4] +=1
#     if data==4:
#         if domain_labels[i] ==0:
#             count[4,0] +=1
#         elif domain_labels[i] == 1:
#             count[4,1] +=1
#         elif domain_labels[i]==2 :
#             count[4,2] +=1
#         elif domain_labels[i] == 3:
#             count[4,3] +=1
#         elif domain_labels[i]==4 :
#             count[4,4] +=1
# print(count)