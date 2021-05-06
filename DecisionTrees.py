import numpy as np
import pyreadr
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from numpy import genfromtxt
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import time
import math
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

allStudentData = genfromtxt('./idlist.csv', delimiter=',')
#Because R likes not a number.
allStudentData = allStudentData[1:]

class CipherData:

    def __init__(self, allStudentData):
        self.dataSet = allStudentData

    def makeDisjunct(self):
        self.labels = self.dataSet[:, :1].astype(float)
        self.flatLabels = np.concatenate(self.labels).ravel()
        self.attributes = self.dataSet[:, 1:].astype(float)

        length = int((len(self.attributes) / 2))

        self.instanceLabels = self.flatLabels.copy()
        self.instanceAttri = self.attributes.copy()

        self.instanceLabelTest = self.instanceLabels[:length]
        self.instanceLabelTrain = self.instanceLabels[length:]
        self.instanceAttriTest = self.instanceAttri[:length]
        self.instanceAttriTrain = self.instanceAttri[length:]

    def makeAllPersonsIn(self):
        allPersonsInDataset = self.dataSet.copy()

        np.random.shuffle(allPersonsInDataset)

        # Extract the labels, convert from string to float, flatten to a vector.
        self.labels = allPersonsInDataset[:, :1].astype(float)
        self.flatLabels = np.concatenate(self.labels).ravel()
        self.attributes = allPersonsInDataset[:, 1:].astype(float)

        length = int((len(self.attributes) / 2))

        self.instanceLabels = self.flatLabels.copy()
        self.instanceAttri = self.attributes.copy()

        self.instanceLabelTest = self.instanceLabels[:length]
        self.instanceLabelTrain = self.instanceLabels[length:]
        self.instanceAttriTest = self.instanceAttri[:length]
        self.instanceAttriTrain = self.instanceAttri[length:]


data = CipherData(allStudentData)

# Extract one persons cipher data
onePersonLabels = data.dataSet[:4000, :1].astype(float)
onePersonFlatLabels = np.concatenate(onePersonLabels).ravel()
onePersonAttributes = data.dataSet[:4000 , 1:].astype(float)

# Compute the first 5 PCA components based on his data.
pca = PCA(n_components=5)
component = pca.fit(onePersonAttributes)
transformedData = component.transform(onePersonAttributes)

print(data)