import pandas as pd
from sklearn.svm._libsvm import predict

iris = pd.read_csv("D:/data lesson 4 glass/Python_Lesson4/glass.csv")
print(iris)
#to import varios libraries

from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split



iris = datasets.load_iris()

# to provide labels for x and y axis x - features and y will be labels
X = iris.data
y = iris.target

# splitting  X and Y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)

# the confusion matrix is created for predictions on SVM
cm = confusion_matrix(y_test, svm_predictions)

print(accuracy)


