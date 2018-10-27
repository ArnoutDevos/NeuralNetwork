import numpy as np
from .layers import *

class serial(object):
    """ Class which allows to define and update layers in a serial way
    
    Init args:
        *args: a series of objects instantiated from the layer module
    """
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
        """Assign parameters to their respective layer
        
        Args:
            name: name of layer
            val: value of the layer weights
        """
        layer_ix = self.paramNamesToIndices[name]
        self.layers[layer_ix].params[name] = val
        
    def assign_grads(self, name, val):
        """Assign gradients to their respective layer
        
        Args:
            name: name of layer
            val: value of the layer gradients
        """
        layer_ix = self.paramNamesToIndices[name]
        self.layers[layer_ix].grads[name] = val
        
    def get_params(self, name):
        """Get parameters from their respective layer
        
        Args:
            name: name of variable, e.g. fc1_w (fully connected layer 1, weight variables)
        Returns:
            Parameter corresponding to the name
        """
        return self.params[name]
    
    def get_grads(self, name):
        """Get gradients from their respective layer
        
        Args:
            name: name of variable, e.g. fc1_w (fully connected layer 1, weight variables)
        Returns:
            Gradient parameter corresponding to the name
        """
        return self.grads[name]
    
    def gather_params(self):
        """Collect the parameters of every layer (layer.params) and put them into the serialized object self.params
        """
        for layer in self.layers:
            for name, value in layer.params.items():
                self.params[name] = value
                
    def gather_grads(self):
        """Collect the gradients of the parameters of every layer (layer.params) and put them into the serialized object self.params
        """
        for layer in self.layers:
            for name, value in layer.params.items():
                self.grads[name] = value
                
def compute_acc(model, data, labels, batch_size = 100):
    """ Compute accuracy for the model
    Args:
        model: model from models module
        data: input features
        labels: golden output labels
        batch_size: how many samples per batch
    
    Returns:
        accuracy of predictions
    """
    preds = predict(model, data)
    accuracy = np.mean(preds == labels)
    return accuracy

def predict(model, data, batch_size = 100):
    """ Compute predictions with the model
    Args:
        model: model from models module
        data: input features
        batch_size: how many samples per batch
    
    Returns:
        predictions for each data point of the data
    """
    labels = np.zeros((data.shape[0],))
    
    preds = []
    
    output = model.forward(data)
    scores = softmax(output)
    pred = np.argmax(scores, axis = 1)
    preds.append(pred)
    preds = np.hstack(preds)
    
    return preds

def batch_iter(y, tx, batch_size=100, num_batches=1, shuffle=False):
    """
    Generate a minibatch iterator for a dataset.
    
    Args:
        y: all output labels
        tx: all input features
        batch_size: size of batch
        num_batches: number of batches
        shuffle: True, randomize the samples selected; False, linear walkthrough of samples
        
    Returns:
        Tuple iterator which contains (y_batch, tx_batch)
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
