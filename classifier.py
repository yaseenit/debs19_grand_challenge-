#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:11:43 2019

@author: yassin
"""
import subprocess

import numpy as np  
#import matplotlib.pyplot as plt  
import pandas as pd 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,Normalizer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
import pickle
from sklearn.metrics import precision_recall_fscore_support



from tpot import TPOTClassifier



#dir = '/home/yassin/debs/vfh/1.3/'
dir = '/home/yassin/debs/objects/esf_objects/'
data = []
for file in sorted(os.listdir(dir)):
  features = np.fromstring([line.rstrip('\n') for line in open(dir+file.strip())][11], dtype=float, sep=' ')
  line = features.tolist()
  if '-' in file:
    line.append(file.strip().split('-')[0].strip())
  else:  
    line.append(file.strip().split('_')[1].strip()) 
  data.append(line)


dataset = pd.DataFrame(data, columns=list(range(640)).append('Class'))

dataset = dataset.drop_duplicates(keep='first')

dataset = dataset.sample(frac=1).reset_index(drop=True)

X = dataset.iloc[:, :-1].values  

y = dataset.iloc[:, 640].values  

classifier = RandomForestClassifier(max_depth=100, n_estimators=50, max_features=10,random_state=3)
#classifier =   GradientBoostingClassifier()

#from sklearn.model_selection import cross_val_score
#from sklearn import metrics
#scores = cross_val_score(classifier, X, y, cv=50, scoring='f1_macro')
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)  
scaler = Normalizer()  
scaler.fit(X_train)


X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

#classifier = KNeighborsClassifier(n_neighbors=20)  
#classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
#classifier.fit(X_train, y_train) 

#tpot = TPOTClassifier(generations=5,verbosity=2,n_jobs=-1)
#tpot.fit(X_train, y_train)

#Best pipeline: 
classifier=   ExtraTreesClassifier(bootstrap=False, criterion='gini', max_features=0.75, min_samples_leaf=2, min_samples_split=6, n_estimators=100)
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)  
#print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

error = []





from sklearn.metrics import accuracy_score, log_loss,recall_score,precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

n_classes = len(np.unique(y_train))

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="rbf", C=0.025, probability=True),
    #NuSVC(probability=True),
    #DecisionTreeClassifier(),
    RandomForestClassifier(),
    RandomForestClassifier(max_depth=100, n_estimators=50, max_features=10,random_state=3),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
#    GaussianMixture(n_components=5,covariance_type='spherical'),
#    GaussianMixture(n_components=10,covariance_type='diag'),
#    GaussianMixture(n_components=50,covariance_type='tied'),
#    GaussianMixture(n_components=100,covariance_type='full')
    ]

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    rec = recall_score(y_test, train_predictions,average='weighted')
    print("Recall: {:.4%}".format(rec))    
    pre = precision_score(y_test, train_predictions,average='weighted')
    print("Precsion: {:.4%}".format(pre)) 
    print(classification_report(y_test, train_predictions))  


  
     










"""








# now you can save it to a file
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f,protocol=2)
    print('classifier has been saved.')

# and later you can load it
#with open('classifier.pkl', 'rb') as f:
#    clf = pickle.load(f)

joblib.dump(scaler, "scaler.save") 
print('scaler has been saved.')    




# Calculating error for K values between 1 and 40
#for i in range(1, 40):  
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    pred_i = knn.predict(X_test)
#    error.append(np.mean(pred_i != y_test))
#    
#plt.figure(figsize=(12, 6))  
#plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
#         markerfacecolor='blue', markersize=10)
#plt.title('Error Rate K Value')  
#plt.xlabel('K Value')  
#plt.ylabel('Mean Error')  

#
#





#dir = '/home/yassin/debs/objects/dataset2_objects/1/'
#for file in os.listdir(dir):
#    file_path = dir+file.strip()+'_vfh.pcd'
#    subprocess.run(["/home/yassin/debs/generate_vfh","1.0",dir+file.strip()])
#    features = np.fromstring([line.rstrip('\n') for line in open(file_path)][11], dtype=float, sep=' ')
#    #features = scaler.transform(features.reshape(1, -1))
#    resultant = classifier.predict(features.reshape(1, -1))[0].strip("'")
#    
#    
#    results = classifier.predict_proba(features.reshape(1, -1))[0]
#
#    # gets a dictionary of {'class_name': probability}
#    prob_per_class_dictionary = dict(zip(classifier.classes_, results))
#    maximum = max(prob_per_class_dictionary, key=prob_per_class_dictionary.get)  # Just use 'min' instead of 'max' for minimum.
#    print(maximum, prob_per_class_dictionary[maximum])
#    if  prob_per_class_dictionary[maximum] > 0.3:
#        os.rename(dir+file.strip(), dir+str(prob_per_class_dictionary[maximum])+maximum+'_'+file.strip())
#    os.remove(file_path)
##    
##    
##
#



#rootdir = '/home/yassin/debs/objects/unlabeled_objects/'
#progress =1 
#for subdir, dirs, files in os.walk(rootdir):
#    for file in files:
#        file_path= os.path.join(subdir, file)
#        subprocess.run(["/home/yassin/debs/generate_vfh","1.0",file_path])
#        features = np.fromstring([line.rstrip('\n') for line in open(file_path+'_vfh.pcd')][11], dtype=float, sep=' ')
#        results = classifier.predict_proba(features.reshape(1, -1))[0]
#        # gets a dictionary of {'class_name': probability}
#        prob_per_class_dictionary = dict(zip(classifier.classes_, results))
#        maximum = max(prob_per_class_dictionary, key=prob_per_class_dictionary.get)
#        if  prob_per_class_dictionary[maximum] > 0.4:
#            os.rename(file_path, '/home/yassin/debs/objects/classified/'+str(prob_per_class_dictionary[maximum])+'_'+maximum+'_'+str(file))
#        os.remove(file_path+'_vfh.pcd')
#        if progress % 1000 == 0:
#          print(progress)
#        progress+=1  
#        
#      




"""
























