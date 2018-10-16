import numpy as np

class fully_connected(object):
    def __init__(self, input_dim, output_dim, init_scale=0.02, name="fc"):
        self.name = name
        self.w_name = name + "w"
        self.b_name = name + "b"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.params = {}
        self.grads = {}
        
        self.params[self.w_name] = init_scale * np.random(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        
        # Variable that stores the current forward pass features
        self.feat = None
        
    def forward(self, feat):
        output = None
        
        x = feat.reshape((feat.shape[0], self.input_dim))
        w = self.params[self.w_name]
        b = self.params[self.b_name]
        
        output = np.dot(x, w) + b
        self.feat = feat
        
        return output
    
    def backward(self, dnext):
        feat = self.feat
        if feat is None:
            raise ValueError("Need to do foward pass first")
        
        dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        batch_size = feat.shape[0]
        input_size = self.input_dim
        output_size = self.output_dim
        
        w = self.params[self.w_name]
        
        dfeat = dnext.dot(w.T).reshape(feat.shape)
        self.grads[self.w_name] = feat.reshape(batch_size, input_size).T.dot(dnext)
        self.grads[self.b_name] = dnext.T.dot(np.ones(batch_size))
        
        return dfeat
    
class relu(object):
    def __init__(self, name="relu"):
        
        self.name = name
        
        self.params = {}
        self.grads = {}
        self.grads[self.name] = None
        self.feat = None
        
    def forward(self, feat):
        output = np.maximum(0, feat)
        self.feat = feat
        return output
    
    def backward(self, dnext):
        feat = self.feat
        if feat is None:
            raise ValueError("Need to do foward pass first")
        
        dfeat = np.array(dnext, copy=true)
        dfeat[feat <= .0] = .0
        
        self.feat = None
        return dfeat