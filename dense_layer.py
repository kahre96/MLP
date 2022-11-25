import numpy as np
import yaml


class DenseLayer(yaml.YAMLObject):
    yaml_tag = u'!Dense_layer'
    def __init__(self, units = 1, activation='relu'):
        self.units = units
        self.activation = activation # f'
        self.weights = None # w
        self.input = None #h(k-1)
        self.a = None # a




    def feed_forward(self,x):
        self.input = x
        self.a = np.dot(x, self.weights)
        if self.activation == "sigmoid":
            # print("using sigmoid")
            return self.sigmoid(self.a)
        if self.activation == "relu":
            # print("using reul")
            return self.relu(self.a)
        if self.activation == "softmax":
            return self.softmax(self.a)

        raise NotImplementedError("chosen Activation doesnt exist for this layer")

    # handles the back prop of this layer
    def back_propagation(self, error, learn_rate):
        # first goes through the derivate of chosen activation function
        if self.activation == "sigmoid":
            s = self.sigmoid_derivate(self.a)
            J = s * error
        elif self.activation == "softmax":
            J = self.softmax_derivate(self.a) * error
        else:
            J = self.relu_derivate(self.a) * error

        # then calculate the errors
        input_error = np.dot(J, self.weights.T)
        weight_error = np.dot(self.input.T, J)

        self.weights -= learn_rate * weight_error

        return input_error

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def sigmoid_derivate(self, x):
        s = self.sigmoid(x) * (1 - self.sigmoid(x))
        return s

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def softmax_derivate(self,x):
        print("no")

    def relu(self, x):
        return x * (x > 0)

    def relu_derivate(self, x):
        return 1. * (x > 0)
