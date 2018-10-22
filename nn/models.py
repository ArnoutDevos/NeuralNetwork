import numpy as np
from .utils import *
from .layers import *


class smallFullyConnected(object):
    
    """
    def __init__(self, dropout_p=0, dtype=np.float32, seed=None):
        self.net = serial(
            FullyConnected(4,30, name="fc1"),
            relu(name="relu1"),
            FullyConnected(30,7, name="fc2"),
            relu(name="relu2")
        )
    
    """
    def __init__(self):
        self.net = serial(
        FullyConnected(30, 30, name = "fc1"),
        relu(name="relu1"),
        FullyConnected(30, 30, name = "fc2"),
        relu(name="relu2"),
        FullyConnected(30, 30, name = "fc3"),
        relu(name="relu3"),
        FullyConnected(30, 2, name="fc4")
        )
    
    def forward(self, feat):
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