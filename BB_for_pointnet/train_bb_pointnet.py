from dataset import *
from InstanceSeg_net import *
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib


lr_steps = [15, 22, 26]
gamma = 0.1
step_index = 0

num_epochs = 30
batch_size = 1
learning_rate = 0.001
num_points = 100000
#network = PointNetDenseCls(k = 2)  #(num_points = num_points)
network = InstanceSeg(num_points = num_points)

network = network.cuda()

train_dataset = InstanceSeg_Dataset(data_path="/home/andyser/data/subt_real",type="train",num_point = num_points)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=16)


regression_loss_func = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(num_epochs):
    network.train()
    for iter, batch in enumerate(train_loader):
        inputs = Variable(batch['x'].cuda())
        labels = Variable(batch['y'].cuda())

        optimizer.zero_grad()	
        outputs = network(inputs)
        #print outputs.shape
        loss = criterion(outputs,labels)
        '''
        outputs = outputs.view(-1,2)
        labels = labels.view(-1)
        #print outputs.shape, labels.shape
        loss = F.nll_loss(outputs, labels)
        '''
        print loss.data
        loss.backward()
        optimizer.step()
    if epoch != 0 and epoch % 5 == 0:
        model_path = "./weights/pointnet_epoch_" + str(epoch)
        torch.save(network.state_dict(), model_path + '.pkl')

    if epoch in lr_steps:
        step_index += 1
        adjust_learning_rate(optimizer, gamma, step_index)