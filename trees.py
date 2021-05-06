import numpy as np
from numpy import genfromtxt
import math
import matplotlib.pyplot as plt
import pyreadr
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree, export_text
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import sys

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

def sortCiphers(data, bucketSize = 400):

	ciphers = {}
	for i in range(0, 10):
		ciphers[i] = []

	for i in range(0, int(len(data) / bucketSize)):
		index = i % 10
		ciphersAtI = data[i * bucketSize:(i * bucketSize) + bucketSize, :]
		cipherArray = ciphers[index]
		if len(cipherArray) == 0:
			cipherArray.append(ciphersAtI)
			ciphers[index] = cipherArray
		else:
			ndArray = cipherArray[0]
			format = np.append(ndArray, ciphersAtI, axis=0)
			ciphers[index] = [format]

	return ciphers



def entropy(index, split, aSum):
	if split[index] == 0:
		return 0
	entropy = -(split[index]/aSum) * math.log(split[index]/aSum)
	return entropy



def calcDecisionPoint(data, pcaCount):
	print(data.shape)
	print(pcaCount)

	bestSplit = 0.5
	splitEntropy = sys.maxsize
	beforeEntropy = sys.maxsize

	for x in range(-10, 10):
		aSplit = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		aSum = 0
		bSplit = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		bSum = 0

		for i in range(0, 4000):
			if data[i][pcaCount] < x/10 :
				aSplit[int(data[i][0])] = aSplit[int(data[i][0])] + 1
				aSum += 1
			else:
				bSplit[int(data[i][0])] = bSplit[int(data[i][0])] + 1
				bSum += 1

		weightA = aSum/(aSum + bSum)
		weightB = bSum/(aSum + bSum)
		sumA = 0
		sumB = 0

		if(aSum == 0 or bSum == 0):
			continue

		for j in range(0,10):
			sumA += entropy(j, aSplit, aSum)
			sumB += entropy(j, bSplit, bSum)

		sumA = sumA * weightA
		sumB = sumB * weightB

		print("split: "+str(x/10)+": "+str(beforeEntropy - (sumA + sumB)) + " entropy: " + str(sumA + sumB))
		beforeEntropy = (sumA + sumB)

		if (sumA + sumB) < splitEntropy:
			splitEntropy = (sumA + sumB)
			bestSplit = x/10


	print(f"Best Split = {bestSplit}, with entropy = {splitEntropy}")
	return bestSplit

def decisionPoint(pcaCount):

	pca5 = PCA(n_components = pcaCount)
	data = allStudentData[:4000, 1:]
	pcaCompoenents = pca5.fit_transform(data)

	myArray = []

	for x in range(0,4000):
		myArray.append(np.insert(pcaCompoenents[x], 0, int(math.floor(x/400))))

	pcaCompoenents = np.array(myArray)

	decisionPoints = []

	for x in range(1,pcaCount+1):
		decisionPoints.append(calcDecisionPoint(pcaCompoenents, x))
	print(decisionPoints)



def plotTreeGraphic(plotTree, fileName):
	#Making the figsize large enough to be readable crashes python
	#and throws a segdump...
	fig = plt.figure(figsize=(50,30))
	_ = plot_tree(plotTree, filled=True)
	fig.savefig(str(fileName)+".png")

def plotTreeText(plotTree, fileName):
	textTree = export_text(plotTree)
	#Printing kinda explodes.
	#print(textTree)
	text_file = open(str(fileName)+".txt", "w")			
	text_file.write(textTree)
	text_file.close()


def decTreeSingleStudent():
	criteria = 'entropy'
	maxDepth = 5

	# Load the data into a CipherData object
	data = CipherData(allStudentData[:4000,:])
	data.makeAllPersonsIn()

	#Define the mode, fit it.
	#Max depth reduces overfitting, 14 seems to be the sweetspot.
	treeClassifier = DecisionTreeClassifier(criterion=criteria, max_depth=maxDepth)
	treeClassifier = treeClassifier.fit(data.instanceAttriTrain, data.instanceLabelTrain)

	#predict on the test data
	predictionTrain = treeClassifier.predict(data.instanceAttriTrain)
	print(f"Accuracy for Decision Tree with criteria={criteria}, max depth={maxDepth}, trainData: {accuracy_score(data.instanceLabelTrain, predictionTrain)}")
	predictions = treeClassifier.predict(data.instanceAttriTest)
	print(f"Accuracy for Decision Tree with criteria={criteria}, max depth={maxDepth}, testData: {accuracy_score(data.instanceLabelTest, predictions)}")
	plotTreeText(treeClassifier, "singleStudent")
	plotTreeGraphic(treeClassifier, "singleStudent")

	pca5 = PCA(n_components = 5)
	treeClassifier = DecisionTreeClassifier(criterion=criteria, max_depth=maxDepth)
	treeClassifier = treeClassifier.fit(pca5.fit_transform(data.instanceAttriTrain), data.instanceLabelTrain)

	predictionTrain = treeClassifier.predict(pca5.fit_transform(data.instanceAttriTrain))
	predictionTest = treeClassifier.predict(pca5.fit_transform(data.instanceAttriTest))

	print(f"PCA Accuracy for Decision Tree with criteria={criteria}, max depth={maxDepth}, trainData: {accuracy_score(data.instanceLabelTrain, predictionTrain)}")
	print(f"PCA Accuracy for Decision Tree with criteria={criteria}, max depth={maxDepth}, testData: {accuracy_score(data.instanceLabelTest, predictionTest)}")

	plotTreeText(treeClassifier, "5pcaSingleStudent")
	plotTreeGraphic(treeClassifier, "5pcaSingleStudent")

def decTreeAllData():
	criteria = 'entropy'
	maxDepth = 5
	pca5 = PCA(n_components=5)
	FoldCounter = 0

	# Load the data into a CipherData object
	data = CipherData(allStudentData)
	data.makeAllPersonsIn()

	accuracy = []
	confusMa = []

	Folds = StratifiedKFold(n_splits=10)
	for train, test in Folds.split(data.attributes, data.labels):
		FoldCounter += 1
		print(f"Running fold #{FoldCounter} for Random Forest")
		# Index our attribtues and labels to get them for train and test in this fold.
		X_train, X_test = data.attributes[train], data.attributes[test]
		y_train, y_test = data.labels[train].ravel(), data.labels[test].ravel()

		# 'Train' our classifier
		treeClassifier = DecisionTreeClassifier(criterion=criteria, max_depth=maxDepth)
		treeClassifier = treeClassifier.fit(X_train, y_train)

		prediction = treeClassifier.predict(X_test)

		# Save the accuracy and confusion matrix for the predictions
		accuracy.append(accuracy_score(y_test, prediction))
		confusMa.append(confusion_matrix(y_test, prediction))

	# Print results
	for x in range(len(accuracy)):
		print(f"Accuracy:\n {accuracy[x]}")


# print(f"confusMa:\n {confusMa[x]}\n")

def decTreeAllDataPCA():
	criteria = 'entropy'
	maxDepth = 5
	pca5 = PCA(n_components = 5)
	FoldCounter = 0

	# Load the data into a CipherData object
	data = CipherData(allStudentData)
	data.makeAllPersonsIn()

	accuracy = []
	confusMa = []

	Folds = StratifiedKFold(n_splits=10)
	for train, test in Folds.split(pca5.fit_transform(data.attributes), data.labels):
		FoldCounter += 1
		print(f"Running fold #{FoldCounter} for Random Forest")
		# Index our attribtues and labels to get them for train and test in this fold.
		X_train, X_test = data.attributes[train], data.attributes[test]
		y_train, y_test = data.labels[train].ravel(), data.labels[test].ravel()

		# 'Train' our classifier
		treeClassifier = DecisionTreeClassifier(criterion=criteria, max_depth=maxDepth)
		treeClassifier = treeClassifier.fit(X_train, y_train)

		prediction = treeClassifier.predict(X_test)

		# Save the accuracy and confusion matrix for the predictions
		accuracy.append(accuracy_score(y_test, prediction))
		confusMa.append(confusion_matrix(y_test, prediction))

	# Print results
	for x in range(len(accuracy)):
		print(f"Accuracy:\n {accuracy[x]}")
		#print(f"confusMa:\n {confusMa[x]}\n")


#This takes a while to run, cut the folds down if testing.
def randForestAllDataCross(splits):
	FoldCounter = 0

	#Haven't messed around with these variables a whole lot yet,
	#but these parameters give quite decent results.
	criteria = 'gini'
	maxDepth = 18
	treeCount = 100

	#Get data
	data = CipherData(allStudentData)
	data.makeAllPersonsIn()

	 # Lists for results.
	accuracy = []
	confusMa = []

	#define our model
	rdForest = RandomForestClassifier(criterion=criteria, max_depth=maxDepth, n_estimators=treeCount)
	pca5 = PCA(n_components=5)
	Folds = StratifiedKFold(n_splits=splits)
	# Folds.split generates indices for the splits.
	for train, test in Folds.split(pca5.fit_transform(data.attributes), data.labels):
		FoldCounter += 1
		print(f"Running fold #{FoldCounter} for Random Forest")
		# Index our attribtues and labels to get them for train and test in this fold.
		X_train, X_test = data.attributes[train], data.attributes[test]
		y_train, y_test = data.labels[train].ravel(), data.labels[test].ravel()

		# 'Train' our classifier
		rdForest.fit(X_train, y_train)

		# Do a prediction on the test data.
		y_pred = rdForest.predict(X_test)

		# Save the accuracy and confusion matrix for the predictions
		accuracy.append(accuracy_score(y_test, y_pred))
		confusMa.append(confusion_matrix(y_test, y_pred))

	# Print results
	for x in range(len(accuracy)):
		print(f"Accuracy:\n {accuracy[x]}")
		#print(f"confusMa:\n {confusMa[x]}\n")

	ret = rdForest.fit(data.attributes, data.labels.ravel())

	return ret

#decisionPoint(5)
#decTreeSingleStudent()
#decTreeAllDataPCA()
#decTreeAllData()
#randForestAllDataCross(10)
