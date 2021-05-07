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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

allStudentData = genfromtxt('./idlist.csv', delimiter=',')
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
testLabels = data.instanceLabelTest


def runMLP(	mlpTrainData, mlpTestData,
			solverType = "adam", activationFunction = "relu", alpha=0.0001, learning_rate = "constant",
			learning_rate_init = 0.001, max_iters = 500, hidden_layer_sizes = (50, 50), applyStandardization = True):
    if applyStandardization:
        scaler = StandardScaler()
        scaler.fit(mlpTrainData)
        mlpTrainData = scaler.transform(mlpTrainData)
        mlpTestData = scaler.transform(mlpTestData)
	
    mlp = MLPClassifier(
		solver=solverType,
		activation = activationFunction,
		alpha = alpha,
		learning_rate = learning_rate,
		learning_rate_init = learning_rate_init,
		max_iter = max_iters,
		hidden_layer_sizes = hidden_layer_sizes
	)

    mlp.fit(mlpTrainData, trainLabels);
    prediction = mlp.predict(mlpTestData)

    print(f"Precision:  {precision_score(data.instanceLabelTest, prediction, average='weighted')}")
    print(f"Accuracy: {accuracy_score(data.instanceLabelTest, prediction)}")
    print(f"confusion Matrix:\n {confusion_matrix(data.instanceLabelTest, prediction)}\n")

	
def runTenserflowConvu():
	train_images = np.reshape(trainData, (20000, 18, 18))
	train_labels = trainLabels
	test_images = np.reshape(data.instanceAttriTest, (20000, 18, 18))
	test_labels = testLabels
	
	print("train image and labels and test image before reshaping:")
	print(train_images.shape)
	print(train_labels.shape)
	print(test_images.shape)
	
	train_images = np.expand_dims(train_images, axis=3)
	test_images = np.expand_dims(test_images, axis=3)

	print("train image and labels and test image after reshaping:")
	print(train_images.shape)
	print(train_labels.shape)
	print(test_images.shape)
	
	num_filters = 8
	filter_size = 3
	pool_size = 2
	
	model = Sequential([
		#layers
		#Three types: Convolutional, max pooling, softmax
		Conv2D(num_filters, filter_size, input_shape=(18, 18, 1)),
		Conv2D(num_filters, filter_size),
		Conv2D(num_filters, filter_size),
		MaxPooling2D(pool_size=pool_size),
		Flatten(),
		Dense(10, activation='softmax'),
	])
	
	model.compile(
		'adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'],
	)
	
	model.fit(
		train_images,
		to_categorical(train_labels),
		epochs=3,
		validation_data=(test_images, to_categorical(trainLabels)),
	)
	
	predictions = model.predict(test_images[:5]) #Testing on the first 5 images
	
	#Print against first 5 labels
	print(np.argmax(predictions, axis=1))
	print(test_labels[:5])
	
runTenserflowConvu()
	
	
	
	
	
	
#runMLP(trainData, data.instanceAttriTest)
	
#hidden_layer_sizes: the ith element represents the number of neurons in the ith hidden layer

#activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
#Activation function for the hidden layer.
#‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
#‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
#‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
#‘relu’, the rectified linear unit function, returns f(x) = max(0, x)	

#solver: solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
#The solver for weight optimization.
#‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
#‘sgd’ refers to stochastic gradient descent.
#‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

#alpha: L2 penalty (regularization term) parameter

#batch_size: default: Auto. size of minibatches for sthocastic optimizers

#learning_rate: constant, invscaling, adaptive. 
#constant is constant learning rate, invscaling gradually decreases learning rate each time step t
#using an inverse scaling exponent of power_t. adaptive keeps learning rate constant to learning_rate_init
#as long as training loss keeps decreasing.

#power_t, used for invscaling (change if needed)

#max_iter: default:200 - max number of iterations if convergence is not reached.

#shuffle: bool: whether to shuffle samples in each iteration: only applicable to solver sgd or adam

#random_state: determines random number gen for weights and b ias initialization

#tol:default: 1e-4 tolerance for optimzation - determines when convergence is reached as a function of the improvement per step

#verbose: bool	

#warm_start: bool. reuse solution of previous call to fit as initialization or just erase

#momentum: default: 0.9. momentum for gradient descent update (between 0 and 1) only for sgd

#nesterovs_momentum: bool. only applicable for sgd and momentym > 0. Wheteher to use nesterovs momentum (whatever that is)

#early_stopping: bool. Whether to terminate training early when validation score is not improving.

#validation_fraction: default 0.1: proportion of training data set aside as validation for early stopping (must be between 0 and 1)

#beta_1: default: 0.9. exponential decay rate for estiamtes of first moment vector in adam (0, 1)

#beta_2: same as 1 lol

#epsilon: default: 1e-8. value for numerical stability in adam.

#n_iter_no_change: deafult = 10. max number of epochs to not meet tol improvement. only effective with sgd or adam

#max_fun: default 15000. only for lbfgs. max number of loss function calls. 







