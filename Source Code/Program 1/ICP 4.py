#importing different libraries to read the CSV file and load the data
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
#to load the data from the file
train_df = pd.read_csv('D:/data lesson 4 glass/Python_Lesson4/train.csv')
# Displays the first five rows of the dataset
train_df.head()
print(train_df.head())
train_df.isnull().sum()
print(train_df.isnull().sum())
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting data to get the correlation
sns.catplot(x ="Sex", hue ="Survived", kind ="count", data = train_df)
print(sns.catplot(x ="Sex", hue ="Survived", kind ="count", data = train_df))

