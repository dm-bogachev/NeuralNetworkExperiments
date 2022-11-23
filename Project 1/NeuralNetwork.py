import numpy as np
from numpy.lib.npyio import load
from numpy.lib.utils import who
import scipy.special


class NeuralNetwork:
    def __init__(self,
                 input_nodes_size,
                 hidden_nodes_size,
                 output_nodes_size,
                 learning_rate,
                 weight_random_function=lambda x, y: (
                     np.random.rand(x, y)-0.5),
                 activation_function=lambda x: scipy.special.expit(x)):
        # Number of nodes in input, hidden and output layers
        self.ins = input_nodes_size
        self.hns = hidden_nodes_size
        self.ons = output_nodes_size
        # Learning rate
        self.lr = learning_rate
        # Weights randomization function
        self.wrf = weight_random_function
        # Activation function
        self.activation_function = activation_function
        #
        # Weights matrices
        # wih: Input -> Hidden
        # who = Hidden -> Output
        # w_i_j - matrix element
        self.wih = self.wrf(self.hns, self.ins)
        self.who = self.wrf(self.ons, self.hns)
        # Aux
        self.wih_filename = 'wih.txt'
        self.who_filename = 'who.txt'

    def train(self, inputs_list, targets_list):
        # Convert inputs and targets into array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        #
        # Forward processing
        #
        # Calculate hidden layer inputs - X = W*I
        hidden_layer_inputs = np.dot(self.wih, inputs)
        # Calculate hidden layer outputs (inputs for output layer) - H = sigmoid(X)
        hidden_layer_outputs = self.activation_function(hidden_layer_inputs)
        # Calculate output layer inputs - X = W*H
        output_layer_inputs = np.dot(self.who, hidden_layer_outputs)
        # Calculate hidden layer outputs - O = sigmoid(X)
        output_layer_outputs = self.activation_function(output_layer_inputs)
        #
        # Backward processing
        #
        # Calculate errors
        output_layer_errors = targets-output_layer_outputs
        # Calculate hidden layer errors
        hidden_layer_errors = np.dot(self.who.T, output_layer_errors)
        # Update weights
        self.who += self.lr*np.dot(output_layer_errors*output_layer_outputs*(1-output_layer_outputs),
                                    np.transpose(hidden_layer_outputs))
        self.wih += self.lr*np.dot(hidden_layer_errors*hidden_layer_outputs*(1-hidden_layer_outputs),
                                    np.transpose(inputs))

    def query(self, inputs_list, as_array=False):
        # Convert inputs into array
        inputs = np.array(inputs_list, ndmin=2).T
        # Calculate hidden layer inputs - X = W*I
        hidden_layer_inputs = np.dot(self.wih, inputs)
        # Calculate hidden layer outputs (inputs for output layer) - H = sigmoid(X)
        hidden_layer_outputs = self.activation_function(hidden_layer_inputs)
        # Calculate output layer inputs - X = W*H
        output_layer_inputs = np.dot(self.who, hidden_layer_outputs)
        # Calculate hidden layer outputs - O = sigmoid(X)
        output_layer_outputs = self.activation_function(output_layer_inputs)

        if as_array:
            return output_layer_outputs
        else:
            return np.argmax(output_layer_outputs)

    def load_weights(self):
        self.wih = np.loadtxt(self.wih_filename, delimiter=',')
        self.who = np.loadtxt(self.who_filename, delimiter=',')

    def save_weights(self):
        np.savetxt(self.wih_filename, self.wih, delimiter=',', fmt='%.15f')
        np.savetxt(self.who_filename, self.who, delimiter=',', fmt='%.15f')

    def show_weights(self):
        print('Weights from Input to Hidden layers')
        print(self.wih)
        print('Weights from Hidden to Output layers')
        print(self.who)

    def clear_weights(self):
        self.wih = self.wrf(self.hns, self.ins)
        self.who = self.wrf(self.ons, self.hns)
        self.save_weights()
