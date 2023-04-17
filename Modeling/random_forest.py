#Implementing random forest over tensorflow mnist
# Path: Modeling\random_forest.py

from tensorflow.keras.datasets import mnist

import numpy as np

from sklearn.ensemble import RandomForestClassifier

class RandomForest_MNIST:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)

    def train(self, x_train, y_train, epochs = None, steps_per_epoch=None):
        #select only steps_per_epoch number of random matching samples
        if steps_per_epoch is not None:
            indices = np.random.choice(x_train.shape[0], steps_per_epoch, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]
        else:
            steps_per_epoch = x_train.shape[0]

        #reshape the data to fit the model
        x_train = x_train.reshape(x_train.shape[0], 28*28)
        #train the model
        self.model.fit(x_train, y_train)
    
    def test(self, x_test, y_test):
        x_test = x_test.reshape(x_test.shape[0], 28*28)
        test_acc = self.model.score(x_test, y_test)
        return -1,test_acc
    
    def predict(self, x):
        x = x.astype(np.float32)
        x = x.reshape(x.shape[0], 28*28)
        predictions = self.model.predict(x)
        return predictions
    
    def display(self):
        """displaying the model graphically
        """
        #display the model on screen
        from sklearn.tree import export_graphviz
        import graphviz
        dot_data = export_graphviz(self.model.estimators_[0], out_file=None, 
                                feature_names=None,  
                                class_names=None,  
                                filled=True, rounded=True,  
                                special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("model")
        graph.view()

    def save(self, path):
        pass

    def load(self, path):
        pass