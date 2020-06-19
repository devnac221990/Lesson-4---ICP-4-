import pandas as pd

df = pd.read_csv("D:\data lesson 4 glass\Python_Lesson4\glass.csv")
print(df)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#this will help in displaying the top 5 values from the dataset
df.head()
print(df.head())


df.shape
df.info()
df.describe()

print(df.describe)

df.isnull().sum()

nb = GaussianNB() # Create a Naive Bayes object

x = df.drop(columns=['Type'])# variables x and y are created
y = df['Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4) #to create train and test data

nb.fit(x_train, y_train)#training part of the model
#Predict testing set
y_pred = nb.predict(x_test)

print(accuracy_score(y_test, y_pred)) #performing the accuracy check for the model

