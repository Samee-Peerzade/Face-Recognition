from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#step1: Import the data
olivetti_data = fetch_olivetti_faces()

# there are 400 images - 10x40 (40 people - 1 person has 10 images) - 1 image = 64x64 pixels
features = olivetti_data.data


# we represent target variables (people) with integers (face ids)
targets = olivetti_data.target

print(features.shape)
print('\n')
print(features)

print(targets)

print(f'\n{targets.shape}')    # 1 person= 10 image = 40*10=400 images

#step2: Visualize
fig,sub_plots= plt.subplots(nrows=5,ncols=8,figsize=(14,8))
fig,sub_plots= plt.subplots(nrows=5,ncols=8,figsize=(14,8))
sub_plots=sub_plots.flatten()

for unique_user_id in np.unique(targets):
    image_index= unique_user_id*8
    sub_plots[unique_user_id].imshow(features[image_index].reshape(64,64), cmap='gray')
    sub_plots[unique_user_id].set_xticks([])
    sub_plots[unique_user_id].set_yticks([])
    sub_plots[unique_user_id].set_title('face id:%s' % unique_user_id)

plt.suptitle("The dataset (40 people)")
plt.show()
print(unique_user_id)

fig,sub_plots= plt.subplots(nrows=1,ncols=10,figsize=(18,9))

for j in range(10):
    
    sub_plots[j].imshow(features[j].reshape(64,64), cmap='gray')
    sub_plots[j].set_xticks([])
    sub_plots[j].set_yticks([])
    sub_plots[j].set_title('face id=0')


plt.show()
# step3: Train and test data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)
print(len(X_train))
print(len(X_test))


# let's try to find the optimal number of eigenvectors (principle components)
pca = PCA()
pca.fit(features)
plt.figure(1,figsize=(12,8))
plt.plot(pca.explained_variance_, linewidth=3)
plt.xlabel('components')
plt.ylabel('Explained_variance')
plt.show()


pca = PCA(n_components=100, whiten=True)
pca.fit(X_train)
X_pca = pca.fit_transform(features)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(features.shape)
print(X_train_pca.shape)

# after we find 100 optimal components we can check the Eigen_faces
# 1 principle components ( eigen vector ) has 4096 features 

number_of_eigenfaces=len(pca.components_)
eigen_faces=pca.components_.reshape((number_of_eigenfaces, 64, 64))
print(number_of_eigenfaces)

matrix=np.cov(X_train)
print(matrix)
matrix.shape
print(len(matrix))
np.mean(X_train)
print(f'Orignal data,:', features.shape)
print('\n')
print(f'Reduced features by PCA:', X_train_pca.shape)

#step4: Eigen faces
fig,sub_plots= plt.subplots(nrows=10,ncols=10,figsize=(15,15))
sub_plots =sub_plots.flatten()

for i in range(number_of_eigenfaces):
    
    sub_plots[i].imshow(eigen_faces[i], cmap='gray')
    sub_plots[i].set_xticks([])
    sub_plots[i].set_yticks([])
    

plt.suptitle("eigenfaces")
plt.show()

#step5:Accuracy Prediction
models = [("Logistic Regression", LogisticRegression()), ("Support Vector Machine", SVC()), ("Naive Bayes Classifier", GaussianNB())]

for name, model in models:

  classifier_model=model
  classifier_model.fit(X_train_pca, y_train)
  
  y_predicted=classifier_model.predict(X_test_pca)


  print("Results with %s:" % name,)
  print("ACCURACY SCORE:%s\n" % (metrics.accuracy_score(y_test, y_predicted)*100) )

