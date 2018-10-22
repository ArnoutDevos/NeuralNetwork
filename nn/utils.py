import numpy as np
from .layers import *

class serial(object):
    def __init__(self, *args):
        
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramNamesToIndices={}
        self.layer_names = {}
        
        layer_ix = 0
        for layer in args:
            for name, value in layer.params.items():
                if value is None:
                    continue
                self.params[name] = value
                self.paramNamesToIndices[name] = layer_ix
            
            for name, value in layer.grads.items():
                self.grads[name] = value
            
            if layer.name in self.layer_names:
                raise ValueError("Layer with same name already exists: "+str(layer.name))
            
            self.layer_names[layer.name] = True
            self.layers.append(layer)
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
            for name, value in layer.params.items():
                self.params[name] = value
                
    def gather_grads(self):
        for layer in self.layers:
            for name, value in layer.params.items():
                self.grads[name] = value
                
def compute_acc(model, data, labels, batch_size = 100):
    preds = predict(model, data)
    #print(preds[0:20])
    #print(labels[0:20])
    accuracy = np.mean(preds == labels)
    return accuracy

def predict(model, data, batch_size = 100):
    labels = np.zeros((data.shape[0],))
    
    preds = []
    #i=0
    """
    for _, data_batch in batch_iter(labels, data, batch_size, num_batches=int(data.shape[0]/batch_size)+1, shuffle=False):
        #print("batch {} predicted out of {}".format(i,data.shape[0]/batch_size))
        output = model.forward(data_batch)
        scores = softmax(output)
        pred = np.argmax(scores, axis = 1)
        preds.append(pred)
        #i+=1
    preds = np.hstack(preds)
    """
    
    output = model.forward(data)
    scores = softmax(output)
    pred = np.argmax(scores, axis = 1)
    preds.append(pred)
    preds = np.hstack(preds)
    
    return preds

def batch_iter(y, tx, batch_size=100, num_batches=1, shuffle=False):
    """
    Generate a minibatch iterator for a dataset.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
