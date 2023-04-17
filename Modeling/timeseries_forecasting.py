#importing sample timeseries data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras


class LSTM_Timeseries:
    def __init__(self, input_shape, output_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(output_shape, activation='softmax')
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

    def display(self):
        return self.model.summary()

def main():
    #csv data from https://www.kaggle.com/rakannimer/air-passengers
    #load the data
    df = pd.read_csv('data/AirPassengers.csv', index_col='Month', parse_dates=True)
    print(df.head())
    #convert the data to monthly frequency
    df.index.freq = 'MS'
    #plot the data
    df.plot(figsize=(12, 6))
    plt.show()

    #split the data into train and test
    train_data = df.iloc[:len(df)-12]
    test_data = df.iloc[len(df)-12:]

    #scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    #create a timeseries generator
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    length = 12
    batch_size = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)

    #create the model
    model = LSTM_Timeseries(input_shape=(length, 1), output_shape=1)
    model.train(scaled_train, scaled_train, epochs=5, steps_per_epoch=1)
    model.display()

    #evaluate the model
    loss, acc = model.test(scaled_test, scaled_test)
    print("Loss: ", loss)
    print("Accuracy: ", acc)

    #plot predictions against expected values
    test_predictions = []
    first_eval_batch = scaled_train[-length:]
    current_batch = first_eval_batch.reshape((1, length, 1))
    for i in range(len(test_data)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    true_predictions = scaler.inverse_transform(test_predictions)
    test_data['Predictions'] = true_predictions
    test_data.plot(figsize=(12,6))
    plt.show()

    return

if __name__ == "__main__":
    main()
