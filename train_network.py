import torch
from torch.optim import Adam
import time 
from statistics import mean
import torch.nn.functional as F
import torch.nn as nn

def loss_evolution(loss_list):
    avg_loss = mean(loss_list)
    prev_avg_loss = mean(loss_list[:-1])

    raw_change = prev_avg_loss - avg_loss
    percentage_change = raw_change / avg_loss * 100

    return avg_loss, percentage_change

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def train_model(unet, trainLoader, testLoader, num_epochs=10, learning_rate=0.01, device=None, loss="mse"):

    if device == "mps":
        DEVICE = torch.device('mps')
        unet = unet.to(DEVICE)
    elif device == "cuda:0":
        DEVICE = torch.device('cuda:0')
        unet = unet.to(DEVICE)
    else:
        DEVICE = torch.device('cpu')
    
    if loss == "BCEwithLogits":
        lossFunc = torch.nn.BCEWithLogitsLoss()
    elif loss == "BCE":
        lossFunc = torch.nn.BCELoss()
    elif loss == "dice":
        lossFunc = DiceLoss()
    else:
        lossFunc = torch.nn.MSELoss()
    opt = Adam(unet.parameters(), lr=learning_rate)

    print("[INFO] training the network...")
    print('num epochs:',num_epochs,type(num_epochs))
    print('learning rate:',learning_rate)
    print('loss:',loss)
    print('training images:',len(trainLoader))
    print('testing images:',len(testLoader))

    #device = torch.device(DEVICE)

    for e in range(num_epochs):
        start_time = time.time()
        unet.train() # set the model in training mode

        # initialize the total training and validation loss
        train_loss_per_epoch = []
        test_loss_per_epoch = []

        # iterating through training data
        for (i, (x, y)) in enumerate(trainLoader):
            x = x.type(torch.float32)
            y = y.type(torch.float32)
            (x,y) = (x.to(DEVICE), y.to(DEVICE)) # sending the data to the device (cpu or GPU)
            pred = unet(x) # make a prediction
            loss = lossFunc(pred, y) # calculate the loss of that prediction
            opt.zero_grad() # zero out the accumulated gradients
            loss.backward() # backpropagate the loss
            opt.step() # update model parameters
            train_loss_per_epoch.append(loss.detach().item())
        
        # switch off autograd
        with torch.no_grad():
            unet.eval() # set the model in evaluation mode
            # iterating through validation data
            for (x, y) in testLoader:
                x = x.type(torch.float32)
                y = y.type(torch.float32)
                (x,y) = (x.to(DEVICE), y.to(DEVICE)) # sending the data to the device (cpu or GPU)
                pred = unet(x) # make a prediction
                loss = lossFunc(pred, y) # calculate the loss of that prediction
                test_loss_per_epoch.append(loss.detach().item())
        
        time_taken = time.time() - start_time
        remaining_time = time_taken * (num_epochs - e - 1)
               
        # calculate the average training and validation loss
        if e == 0:
            #train_loss_per_epoch = np.array(train_loss_per_epoch)
            avg_train_loss = mean(train_loss_per_epoch)
            avg_test_loss = mean(test_loss_per_epoch)
            print("[INFO] EPOCH: {}/{}  Train loss: {:.6f}  Test loss: {:.6f}  Time taken: {:.0f}s  Remaining time: {:.0f}s".format(e + 1, num_epochs, avg_train_loss, avg_test_loss, time_taken, remaining_time))
            #TODO: print dice loss
        else:
            avg_train_loss, change_train_loss = loss_evolution(train_loss_per_epoch)
            avg_test_loss, change_test_loss = loss_evolution(test_loss_per_epoch)
            print("[INFO] EPOCH: {}/{}  Train loss: {:.6f}  Test loss: {:.6f}  Train loss change: {:.2f}  Test loss change: {:.2f}  Time taken: {:.0f}s  Remaining time: {:.0f}s".format(e + 1, num_epochs, avg_train_loss, avg_test_loss, change_train_loss, change_test_loss, time_taken, remaining_time))
    
    return unet