import deeplearning, random_forest, timeseries_forecasting

from tensorflow.keras.datasets import mnist

import numpy as np

def main():
    model_list = \
        [
            # deeplearning.LSTM_MNIST(),
            random_forest.RandomForest_MNIST()
        ]
    
    x_train, y_train = mnist.load_data()[0]
    x_test, y_test = mnist.load_data()[1]
    
    for model in model_list:
        model.train(x_train, y_train, epochs=1,steps_per_epoch=1000)
        test_loss, test_acc = model.test(x_test, y_test)
        print('Test accuracy:', test_acc)
        model.save("model.h5")
        model.load("model.h5")
        predictions = model.predict(x_test)

        # randomizing displayed test sample
        index = np.random.randint(0, x_test.shape[0])
        #print the class name
        print(model.__class__.__name__)
        print(predictions[index])
        print(np.argmax(predictions[index]))
        print(y_test[index])

        model.display()

    return

if __name__=="__main__":
    main()