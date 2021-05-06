import numpy as np
import pyreadr
import random
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

allStudentData = np.genfromtxt('./idlist.csv', delimiter=',')
# Because R likes not a number.
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
    scaleAccuracy = []
    scalePrecision = []
    autoAccuracy = []
    autoPrecision = []

    for coefficient in gammaCoefficients:

        for c in cs:
            linearSVM = svm.SVC(kernel="linear", C=c, gamma=coefficient)
            linearSVM.fit(trainData, trainLabels)

            prediction = linearSVM.predict(data.instanceAttriTest)

            print("GammeCoefficient = " + str(coefficient))
            print("C = " + str(c))
            if coefficient == 'scale':
                scaleAccuracy.append(accuracy_score(data.instanceLabelTest, prediction))
                scalePrecision.append(precision_score(data.instanceLabelTest, prediction, average='weighted'))
            else:
                autoAccuracy.append(accuracy_score(data.instanceLabelTest, prediction))
                autoPrecision.append(precision_score(data.instanceLabelTest, prediction, average='weighted'))
        if coefficient == 'scale':
            print("Accuracy: " + str(scaleAccuracy))
            print("Precision: " + str(scalePrecision))
        else:
            print("Accuracy: " + str(autoAccuracy))
            print("Precision: " + str(autoPrecision))

    scaleAccuracy = []
    scalePrecision = []
    autoAccuracy = []
    autoPrecision = []

    print("****** KERNEL RBF ******")
    for coefficient in gammaCoefficients:
        for c in cs:
            rbfSVM = svm.SVC(kernel="rbf", C=c, gamma=coefficient)
            rbfSVM.fit(trainData, trainLabels)

            prediction = rbfSVM.predict(data.instanceAttriTest)

            print("GammeCoefficient = " + str(coefficient))
            print("C = " + str(c))
            if coefficient == 'scale':
                scaleAccuracy.append(accuracy_score(data.instanceLabelTest, prediction))
                scalePrecision.append(precision_score(data.instanceLabelTest, prediction, average='weighted'))
            else:
                autoAccuracy.append(accuracy_score(data.instanceLabelTest, prediction))
                autoPrecision.append(precision_score(data.instanceLabelTest, prediction, average='weighted'))
        if coefficient == 'scale':
            print("Accuracy: " + str(scaleAccuracy))
            print("Precision: " + str(scalePrecision))
        else:
            print("Accuracy: " + str(autoAccuracy))
            print("Precision: " + str(autoPrecision))

    scaleAccuracy = []
    scalePrecision = []
    autoAccuracy = []
    autoPrecision = []

    print("****** KERNEL Polynomial ******")
    for coefficient in gammaCoefficients:
        for c in cs:
            polynomialSVM = svm.SVC(kernel="poly", degree=3, C=c, gamma=coefficient)
            polynomialSVM.fit(trainData, trainLabels)

            prediction = polynomialSVM.predict(data.instanceAttriTest)

            print("GammeCoefficient = " + str(coefficient))
            print("C = " + str(c))
            if coefficient == 'scale':
                scaleAccuracy.append(accuracy_score(data.instanceLabelTest, prediction))
                scalePrecision.append(precision_score(data.instanceLabelTest, prediction, average='weighted'))
            else:
                autoAccuracy.append(accuracy_score(data.instanceLabelTest, prediction))
                autoPrecision.append(precision_score(data.instanceLabelTest, prediction, average='weighted'))
        if coefficient == 'scale':
            print("Accuracy: " + str(scaleAccuracy))
            print("Precision: " + str(scalePrecision))
        else:
            print("Accuracy: " + str(autoAccuracy))
            print("Precision: " + str(autoPrecision))


def runMLP():
    # We use Stochastic Gradient Descent to be able to get learning errors, and
    # Adam is more suited for larger datasets. Lbfgs would be better than sgd but
    # does not provide error information on learning.
    solvers = ["lbfgs", "sgd", "adam"]
    # We disregard the tanh activation function: It ranges from -1 to 1, mapping inputs
    # Distinctly towrads -1 when negative, and 0 inputs close to 0. Behavior making it
    # well suited for classification between two classes, thus not our choice.
    # For much the same reason we disregard the logistic (sigmoid) activation function.
    # We use relu for our activation function. It goes from 0 to infinity and can suffer from 'dying'
    # when the input includes negative values. Not a great concern for us. It has the advantage
    # of not suffering from the vanishing gradient problem (while a benefit over of activations such
    # as sigmoid and tanh, whom both suffer from this problem,
    # it would be more relevant if our model included a greater number of layers).
    # The fact that it sets all negative inputs to 0 also helps produce a more sparse model,
    # silencing low/non importance neurons.
    activationFunction = ["identity", "logistic", "tanh", "relu"]
    # We leave our alpha value as default. Could be optimised further with trial and error.
    alpha = 0.001
    # We set our learning rate to adaptive, a perk of using a gradient descent algorithm.
    learning_rate = ["constant", "invscaling", "adaptive"]
    learning_rate_init = 0.001
    # To avoid needles learning we set a max iter of 200.
    max_iters = 200
    hidden_layer_sizes = (50, 10)
    applyStandardization = True
    mlpTrainData = trainData;
    mlpTestData = data.instanceAttriTest

    if applyStandardization:
        scaler = StandardScaler()
        scaler.fit(mlpTrainData)
        mlpTrainData = scaler.transform(mlpTrainData)
        mlpTestData = scaler.transform(mlpTestData)

    mlp = MLPClassifier(
        solver="sgd",
        activation="relu",
        alpha=alpha,
        learning_rate="adaptive",
        max_iter=max_iters,
        learning_rate_init=learning_rate_init,
        hidden_layer_sizes=hidden_layer_sizes
    )

    mlp.fit(mlpTrainData, trainLabels);
    print(mlp.loss_curve_)

    t_sizes, t_scores, valid_scores = learning_curve(mlp, mlpTrainData, trainLabels,
                                                     train_sizes=np.linspace(0.1, 1.0, 5), cv=5,
                                                     scoring='neg_mean_squared_error', error_score='raise')

    prediction = mlp.predict(mlpTestData)

    print(f"Precision:  {precision_score(data.instanceLabelTest, prediction, average='weighted')}")
    print(f"Accuracy: {accuracy_score(data.instanceLabelTest, prediction)}")
    print(f"confusion Matrix:\n {confusion_matrix(data.instanceLabelTest, prediction)}\n")

    fig, axs = plt.subplots(3)
    axs[0].plot(np.arange(0, mlp.n_iter_, 1), mlp.loss_curve_, label='')
    axs[0].set_title("Loss Curve")
    axs[1].plot(t_sizes, t_scores.mean(axis=1), )
    axs[1].set_title("Training Scores")
    axs[2].plot(t_sizes, valid_scores.mean(axis=1), )
    axs[2].set_title("Validation Scores")
    plt.ylabel('Error/Score')
    plt.xlabel('Set Size')
    plt.legend()
    fig.tight_layout()
    plt.show()

runSVM()

#runMLP()