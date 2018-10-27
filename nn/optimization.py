import numpy as np

class SGD:
    """Stochastic gradient descent optimizer object
        Args: 
            net: network of class Serial that needs to be optimized
            lr: learning rate of this vanilla SGD optimizer
    """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr
        
    def step(self):
        """ Take one step of gradient descent with learning rate present in this object, and gradients present inside the network"""
        for layer in self.net.layers:
            for n, v in layer.params.items():
                dv = layer.grads[n]
                layer.params[n] -= self.lr * dv
                
class Adam:
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        self.t +=1
        for layer in self.net.layers:
            for n, v in layer.params.items():
                if not n in self.mt:
                    self.mt[n] = 0
                if not n in self.vt:
                    self.vt[n] = 0
                
                self.mt[n] = self.beta1 * self.mt[n] + (1. - self.beta1) * layer.grads[n]
                self.vt[n] = self.beta2 * self.vt[n] + (1. - self.beta2) * (layer.grads[n]**2)
                
                cmt = self.mt[n] / (1. - self.beta1**self.t)
                cvt = self.vt[n] / (1. - self.beta2**self.t)
                
                layer.params[n] -= self.lr * cmt / (np.sqrt(cvt) + self.eps)