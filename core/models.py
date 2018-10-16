import numpy as np
from layers import *
from utils import *

class smallFullyConnected(object):
    def __init__(self)
        self.net = sequential(
        fc(30000, 200, name = "fc1"),
        relu(name="relu1"),
        fc(200, 1, name="fc2")
        )
    
    def forward(self):
        output = feat
        for layer in self.net.layers:
            output = layer.forward(output)
        
        self.net.gather_params()
        return output
    
    def backward(self, dnext):
        for layer in self.net.layers:
            dnext = layer.backward(output)
        
        self.net.gather_grads()
        return dnext