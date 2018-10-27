import numpy as np
from .utils import *
from .layers import *


class smallFullyConnected(object):
    """ A deep neural network 2 hidden layer fully connected model, specified using the Serial class from the utils module.
    
    Init args:
        input_dim: input dimension, default = 30
        hidden_neurons: how many neurons each hidden layer should have, default = 100
    """
    def __init__(self, input_dim = 30, hidden_neurons = 100):
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        
        self.net = serial(
        FullyConnected(input_dim, hidden_neurons, name = "fc1"),
        relu(name="relu1"),
        FullyConnected(hidden_neurons, hidden_neurons, name = "fc2"),
        relu(name="relu2"),
        FullyConnected(hidden_neurons, 2, name = "fc3")
        )
        
    def reset(self):
        """ Reset allows to reinitialize parameters, without destroying the object itself or the need to recreate it"""
        self.net = serial(
        FullyConnected(self.input_dim, self.hidden_neurons, name = "fc1"),
        relu(name="relu1"),
        FullyConnected(self.hidden_neurons, self.hidden_neurons, name = "fc2"),
        relu(name="relu2"),
        FullyConnected(self.hidden_neurons, 2, name = "fc3")
        )
    
    def forward(self, feat):
        """
        """
        output = feat
        for layer in self.net.layers:
            output = layer.forward(output)
        
        self.net.gather_params()
        return output
    
    def backward(self, dnext):
        for layer in self.net.layers[::-1]:
            dnext = layer.backward(dnext)
        
        self.net.gather_grads()
        return dnext