import numpy as np

class SGD:
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr
        
    def step(self):
        for layer in self.net.layers:
            for n, v in layer.params.items():
                dv = layer.grads[n]
                layer.params[n] -= self.lr * dv