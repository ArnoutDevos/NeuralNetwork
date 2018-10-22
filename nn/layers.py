import numpy as np

class FullyConnected(object):
    def __init__(self, input_dim, output_dim, init_scale=0.02, name="fc"):
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        
        # Variable that stores the current forward pass features
        self.feat = None
        
    def forward(self, feat):
        output = None
        
        assert np.prod(feat.shape[1:]) == self.input_dim, "But got {} and {}".format(
            np.prod(feat.shape[1:]), self.input_dim)
        
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
        
        dfeat = np.dot(dnext, w.T).reshape(feat.shape)
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
        
        dfeat = np.array(dnext, copy=True)
        dfeat[feat <= .0] = .0
        
        self.feat = None
        return dfeat
    
class cross_entropy:
    def __init__(self):
        self.dLoss = None
        self.label = None
        
    def forward(self, feat, label):
        scores = softmax(feat)
        loss = -np.sum(np.log(scores[np.arange(scores.shape[0]), label]))
        loss = loss/label.shape[0]
        
        self.dLoss = scores.copy()
        self.label = label
        
        return loss
    
    def backward(self):
        dLoss = self.dLoss
        if dLoss is None:
            raise ValueError("Should do forward step first!")
        
        dLoss[np.arange(dLoss.shape[0]), self.label] -= 1
        dLoss /= self.label.shape[0]
        
        self.dLoss = dLoss
        return dLoss
    
def softmax(feat):
    feat = (feat.T - np.max(feat, axis = 1)).T # To improve numerical stability
    
    scores = np.exp(feat)
    scores = (scores.T / np.sum(scores, axis=1)).T
    
    return scores