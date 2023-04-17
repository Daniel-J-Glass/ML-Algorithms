#Implementing gradient boosting on MNIST dataset
#
# Path: Modeling\gradientboosting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_digits
from torchvision import datasets, transforms

def main():
    #import mnist data
    mnist = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    X = mnist.data.numpy()
    y = mnist.targets.numpy()
    
    #reshape data
    X = X.reshape(X.shape[0], -1)

    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #create model
    model = GradientBoostingClassifier()

    #train model
    model.fit(X_train, y_train)

    #predict
    y_pred = model.predict(X_test)

    #evaluate
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    #plot confusion matrix
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True)
    plt.show()

    plt.plot(model.feature_importances_)
    plt.show()

    plt.imshow(X[0].reshape(8,8))
    print(model.predict(X[0].reshape(1,-1)))
    print(y[0])
    
if __name__ == "__main__":
    main()