import numpy as np

class serial(object):
    def __init__(self, *args):
        
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramNamesToIndices={}
        self.layer_names = {}
        
        layer_ix = 0
        for layer in args:
            for name, value in layer.params.iteritems():
                if value is None:
                    continue
                self.params[name] = value
                self.paramNamesToIndices[name] = layer_ix
            
            for name, value in layer.grads.iteritems():
                self.grads[name] = value
            
            if layer.name in self.layer_names:
                raise ValueError("Layer with same name already exists: "+str(layer.name))
            
            self.layer_names[layer.name] = True
            self.layers.append{layer}
            layer_ix += 1
    
    def assign_params(self, name, val):
        layer_ix = self.paramNamesToIndices[name]
        self.layers[layer_ix].params[name] = val
        
    def assign_grads(self, name, val):
        layer_ix = self.paramNamesToIndices[name]
        self.layers[layer_ix].grads[name] = val
        
    def get_params(self, name):
        return self.params[name]
    
    def get_grads(self, name):
        return self.grads[name]
    
    def gather_params(self):
        """
        Collect the parameters of every layer (layer.params) and put them into the serialized object self.params
        """
        for layer in self.layers:
            for name, value in layer.params.iteritems():
                self.params[name] = value
                
    def gather_grads(self):
        for layer in self.layers:
            for name, value in layer.params.iteritems():
                self.grads[name] = value