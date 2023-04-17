import tensorflow as tf
from tensorflow import keras

import numpy as np

class LSTM_MNIST:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.LSTM(128, input_shape=(28, 28), return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(128),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
    def train(self, x_train, y_train, epochs, steps_per_epoch=None):
        #select only steps_per_epoch number of random matching samples
        if steps_per_epoch is not None:
            indices = np.random.choice(x_train.shape[0], steps_per_epoch, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]
        else:
            steps_per_epoch = x_train.shape[0]

        #reshape the data to fit the model
        x_train = x_train.reshape(x_train.shape[0], 28, 28)
        #train the model
        self.model.fit(x_train, y_train, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def test(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
        return test_loss, test_acc
    
    def predict(self, x):
        x = x.astype(np.float32)
        predictions = self.model.predict(x)
        return predictions
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)