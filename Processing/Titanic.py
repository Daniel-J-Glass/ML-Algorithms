import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import os

import requests

# downloading titanic dataset csv file from github
# http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls

# import the data
df = pd.read_csv('data/train.csv')

# PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare

# print(df.head())
# print(df.describe())

df = df.drop(['PassengerId','Name','Cabin', 'Ticket'], axis=1)
df['Embarked'] = df['Embarked'].fillna('S')
# print(df['Parch'].value_counts())

#minmax scaling for age and fare
scaler = MinMaxScaler()
df['Survived'] = scaler.fit_transform(df[['Survived']])
df['Age'] = scaler.fit_transform(df[['Age']])
df['Fare'] = scaler.fit_transform(df[['Fare']])

onehot = OneHotEncoder()

df['Pclass'] = onehot.fit_transform(df['Pclass'])
df['Parch'] = onehot.fit_transform(df['Parch'])
df['Sex'] = onehot.fit_transform(df['Sex'])
df['Embarked'] = onehot.fit_transform(df['Embarked'])


print(df.head())
