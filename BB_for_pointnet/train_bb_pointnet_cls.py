from dataset import *
from InstanceSeg_net import *
from pointnet import *
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib


lr_steps = [8, 16, 20]
gamma = 0.1
step_index = 0
class_num = 3
num_epochs = 30
batch_size = 20
num_workers = 8
learning_rate = 0.0001
num_points = 8000
network = PointNetCls(k = class_num, feature_transform = True)  #(num_points = num_points)
#network = InstanceSeg(num_points = num_points)
network = network.cuda()

train_dataset = clsSeg_Dataset(data_path="/home/andyser/data/subt_real",num_point = num_points)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers)
test_dataset = clsSeg_Dataset(data_path="/home/andyser/data/subt_real", image_sets=['test'], num_point = num_points)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=2, shuffle=False,
                                           num_workers=num_workers)
iteration = 0

regression_loss_func = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def cls_test():
    network.eval()
    count = np.zeros(class_num, dtype = np.float128)
    acc = np.zeros(class_num, dtype = np.float128)

    for iter, batch in enumerate(test_loader):
        inputs = Variable(batch['x'].cuda())
        labels = batch['y'].numpy()

        outputs = network(inputs)[0]
        outputs = outputs.data.cpu().numpy()
        for i in range(len(outputs)):
            count[labels[i]] += 1
            if outputs[i].argmax() == labels[i]:
                acc[labels[i]] += 1

            
    print "test acc result: ", acc/count 

    f1 = open("./score/cls_acc.txt","a+")
    f1.write('epoch:'+ str(epoch) + ', acc: ' + str(acc/count) + '\n' )

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    global learning_rate
    lr = learning_rate * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(num_epochs):
    print ("=============== epoch " + str(epoch) + " ===============")
    network.train()
    for iter, batch in enumerate(train_loader):
        inputs = Variable(batch['x'].cuda())
        labels = Variable(batch['y'].cuda())
        labels = labels.view(-1)

        optimizer.zero_grad()	
        outputs = network(inputs)[0]

        loss = F.nll_loss(outputs, labels)
        #loss = criterion(outputs,labels)

        for param_group in optimizer.param_groups:
            print "lr: ", param_group['lr'], "/ loss: ", loss.data
        loss.backward()
        optimizer.step()
    if epoch > 0 and epoch % 2 == 0:
        model_path = "./cls_weights/pointnet_cls_epoch_" + str(epoch)
        torch.save(network.state_dict(), model_path + '.pkl')

    if epoch in lr_steps:
        step_index += 1
        adjust_learning_rate(optimizer, gamma, step_index)

    cls_test()