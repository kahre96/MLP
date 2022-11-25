import numpy as np
import yaml
from sklearn.model_selection import train_test_split


class SequentialModel(object):
    '''A class that manages a certain sequence of layers in a neural network    '''

    def __init__(self, layers=None, loss_func='square_error', learn_rate=0.2):
        if layers is None:
            layers = []
        self.layers = layers
        self.loss_func = loss_func
        self.learn_rate = learn_rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, x_data, y_data, epochs):



        # init random weights if there's no weights
        for index, layer in enumerate(self.layers):
            # make sure not to override weights if a model is loaded
            if layer.weights is not None:
                continue
            if index == 0:
                # first layers uses the data as input
                layer.weights = np.random.rand(x_data.data.shape[2],layer.units) - 0.49
                continue
            layer.weights = np.random.rand(self.layers[index-1].units, layer.units) - 0.49

        '''Goes through each input datapoint in x_data and y_data and perform feed_forward and feed_backward on each of the datapoint.
        Repeat the procedure for epochs times.'''

        for i in range(epochs):
            err = 0
            correct = 0

            # split train and data inside the epoch to get a different train and validation set each epoch
            x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
            samples = len(x_train)

            for j in range(samples):
                x = x_train[j]
                for layer in self.layers:
                    x = layer.feed_forward(x)

                # calculate the error of the prediciton with mse
                err += self.mse(y_train[j], x)


                # check if a datapoint got labeled correctly
                if np.argmax(y_train[j]) == np.argmax(x):
                    correct += 1


                # backpropogation
                error = self.mse_prime(y_train[j], x)
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, self.learn_rate)

            # print out error and accuracy
            err /= samples

            print(f'epoch {i}  error={err}')
            print(f" train accuracy: {correct/samples}")

            # end each epoch with prediction on a validation set
            self.predict(x_val, y_val)


    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true)/y_true.size

    def predict(self, x_data, y_data=None):
        '''Performs predictions on all datapoints in x_data and returns an array of the predictions given by the results of feed_forward'''
        guesses = np.zeros(len(x_data))
        for i, x in enumerate(x_data):
            for layer in self.layers:
                x = layer.feed_forward(x)
            # store index of the predicted label in an array
            guesses[i] = np.argmax(x)

        if y_data is not None:
            y_true = np.zeros(len(y_data))
            for i, y in enumerate(y_data):
                y_true[i] = np.argmax(y)
            correct = np.sum(guesses == y_true)
            acc = correct/len(x_data)
            print(f"validation accuracy: {acc}")
        return guesses

    # save all the values of the model to a yaml file
    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump_all((self.learn_rate, self.loss_func, self.layers), f)

    # loads a model from a yaml file
    def load(self, filename):
        with open(filename, 'r') as f:
            data = list(yaml.unsafe_load_all(f))
            self.learn_rate = data[0]
            self.loss_func = data[1]
            self.layers = data[2]
            print(data)

    # outputs the structure of the model
    def print_model(self):
        for layer in self.layers:
            print(layer.units,"  ", layer.activation)

