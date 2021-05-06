import numpy as np
import pyreadr
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
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
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
data.makeDisjunct()

trainData = data.instanceAttriTrain
trainLabels = data.instanceLabelTrain

# Test the kernel type - Linear Kernel --> RBF Kernel --> Polynomial Kernel
# Test kernel parameters
# Test the cost parameter C


def runSVM():
    cs = [1, 3, 5, 10, 100]
    gammaCoefficients = ['scale', 'auto']

    print("****** KERNEL LINEAR ******")
    for coefficient in gammaCoefficients:
        for c in cs:
            linearSVM = svm.SVC(kernel="linear", C=c, gamma=coefficient)
            linearSVM.fit(trainData, trainLabels)

            prediction = linearSVM.predict(data.instanceAttriTest)

            print("GammeCoefficient = " + str(coefficient))
            print("C = " + str(c))
            print(f"Precision  {precision_score(data.instanceLabelTest, prediction, average='weighted')}")
            print(f"Accuracy: {accuracy_score(data.instanceLabelTest, prediction)}")
            print(f"confusion Matrix:\n {confusion_matrix(data.instanceLabelTest, prediction)}\n")

    print("****** KERNEL RBF ******")
    for coefficient in gammaCoefficients:
        for c in cs:
            rbfSVM = svm.SVC(kernel="rbf", C=c, gamma=coefficient)
            rbfSVM.fit(trainData, trainLabels)

            prediction = rbfSVM.predict(data.instanceAttriTest)

            print("GammeCoefficient = " + str(coefficient))
            print("C = " + str(c))
            print(f"Precision:  {precision_score(data.instanceLabelTest, prediction, average='weighted')}")
            print(f"Accuracy: {accuracy_score(data.instanceLabelTest, prediction)}")
            print(f"confusion Matrix:\n {confusion_matrix(data.instanceLabelTest, prediction)}\n")

    print("****** KERNEL Polynomial ******")
    for coefficient in gammaCoefficients:
        for c in cs:
            polynomialSVM = svm.SVC(kernel="poly", degree=3, C=c, gamma=coefficient)
            polynomialSVM.fit(trainData, trainLabels)

            prediction = polynomialSVM.predict(data.instanceAttriTest)

            print("GammeCoefficient = " + str(coefficient))
            print("C = " + str(c))
            print(f"Precision:  {precision_score(data.instanceLabelTest, prediction, average='weighted')}")
            print(f"Accuracy: {accuracy_score(data.instanceLabelTest, prediction)}")
            print(f"confusion Matrix:\n {confusion_matrix(data.instanceLabelTest, prediction)}\n")


def runMLP():

    solvers =["lbfgs", "sgd", "adam"]
    activationFunction = ["identity", "logistic", "tanh", "relu"]
    alpha=0.0001
    learning_rate = ["constant", "invscaling", "adaptive"]
    learning_rate_init = 0.001
    max_iters = 500
    hidden_layer_sizes = (50, 50)

    for solver in solvers:
        print(solver)

    applyStandardization = True

    mlpTrainData = trainData;
    mlpTestData = data.instanceAttriTest

    if applyStandardization:
        scaler = StandardScaler()
        scaler.fit(mlpTrainData)
        mlpTrainData = scaler.transform(mlpTrainData)
        mlpTestData = scaler.transform(mlpTestData)

    mlp = MLPClassifier(
        solver="lbfgs",
        activation="relu",
        alpha=alpha,
        learning_rate="constant",
        learning_rate_init=learning_rate_init,
        max_iter=max_iters,
        hidden_layer_sizes=hidden_layer_sizes
    )

    mlp.fit(mlpTrainData, trainLabels);
    prediction = mlp.predict(mlpTestData)

    print(f"Precision:  {precision_score(data.instanceLabelTest, prediction, average='weighted')}")
    print(f"Accuracy: {accuracy_score(data.instanceLabelTest, prediction)}")
    print(f"confusion Matrix:\n {confusion_matrix(data.instanceLabelTest, prediction)}\n")

runMLP()






