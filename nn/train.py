from .models import *
from .layers import *
from .utils import *
from .optimization import *

def train_net(labels, data, model, loss_function, optimizer, batch_size, max_epochs, show_every = 50000):
    
    opt_val_acc = .0
    opt_params = None
    
    iters_per_epoch = max(data.shape[0] / batch_size, 1)
    max_iters = iters_per_epoch * max_epochs
    
    loss_hist = []
    train_acc_hist = []
    
    iter = 0
    for epoch in range(max_epochs):
        iter_start = epoch * iters_per_epoch
        iter_end = (epoch + 1) * iters_per_epoch
        
        #for iter in range(iter_start, iter_end):
        for labels_batch, data_batch in batch_iter(labels, data, batch_size=batch_size, num_batches=int(iters_per_epoch), shuffle = True):
            labels_batch = labels_batch.astype(int)
            ## Update the parameters of the network by performing
            # 1. Forward pass
            # 2. Backward pass
            # 3. Optimizer step
            
            # Flatten data for input
            data_batch = data_batch.reshape((data_batch.shape[0],-1))
            
            # Forward pass
            output = model.forward(data_batch)
            loss = loss_function.forward(output, labels_batch)
            
            # Backward
            dLoss = loss_function.backward()
            dX = model.backward(dLoss)
            
            # Optimizer step
            optimizer.step()
            
            # Save loss
            loss_hist.append(loss)
            
            if iter % show_every == 0:
                print("(Iteration {} / {}) loss: {}".format(iter+1, max_iters, loss_hist[-1]))
                
            iter += 1
                
        train_acc = compute_acc(model, data, labels, batch_size=batch_size)
        train_acc_hist.append(train_acc)
        
        opt_params = model.net.params
        
        print("(Epoch {} / {}) Training Accuracy: {}".format(
            epoch+1, max_epochs, train_acc))
    
    return loss_hist, opt_params