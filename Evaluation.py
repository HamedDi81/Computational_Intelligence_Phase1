from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = datasets.load_iris()

x = iris.data
y = iris.target

print(x.shape)
print(y.shape)
#print(np.unique(y))
def purity_score(y_true, y_pred):   #calculate the purity
    cm = contingency_matrix(y_true, y_pred)     #obtaining the contingency matrix
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)     #purity formula
rn,pu = [],[]   #void lists to store the Rand-Index and purity values on each K
x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=42,test_size=0.2)   #split the train and test datas
plt.figure(figsize=[9,9])
for i in range(1,10):
    kmeans = KMeans(n_clusters = i) #build the model
    kmeans.fit(x_train)     #fit the datas
    x_pred = kmeans.predict(x_train)    #predict using the trained model(we could alse use kmean.labels_)
#   print(x_pred.shape)
    rn.append(adjusted_rand_score(y_train,x_pred))  #save each R-Index value
    pu.append(purity_score(y_train,x_pred))         #save each Purity value
kmeans = KMeans(n_clusters = 3)     #retrain the model with desired k ( k= 3)
kmeans.fit(x_train)                 #fit the model
x_pred = kmeans.predict(x_train)    #predict using the trained model(we could alse use kmean.labels_)
plt.plot(np.arange(1,10),rn);           #plot
plt.plot(np.arange(1,10),pu);           #Rand-Index and
plt.xlabel("K");                        #Purity
plt.ylabel("Purity & Rand-Index values")#values
plt.legend(["Rand-Index","Purity"])
cluster_labels = []                     #assign a value to each cluster
for i in np.unique(x_pred):
    c = (x_pred == i)
    label, count = np.unique(y_train[c], return_counts=True)
    cluster_labels.append(label[np.argmax(count)])
print(cluster_labels)               #print the assigned values
test = kmeans.predict(x_test)       #predict the test data
x2 = test.copy()                    #store a copy of predicted test datas
y_pred = np.select([x2 == i for i in range(kmeans.cluster_centers_.shape[0])],cluster_labels,x2) #replace each value of predicted test datas
                                                                                                 #based on assigned values of each cluster
print(f"Accuracy: {accuracy_score(y_test,y_pred) * 100}%")        #calculate the accuracy of the predicted values                              