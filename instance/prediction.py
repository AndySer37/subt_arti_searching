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
import pcl


num_epochs = 10
batch_size = 1
learning_rate = 0.001
num_points = 150000
#network = PointNetDenseCls(k = 2)  #(num_points = num_points)
network = InstanceSeg(num_points = num_points)
network = network.cuda()

test_dataset = InstanceSeg_Dataset(data_path="/home/andyser/data/subt_real",type="train",num_point = num_points)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=1, shuffle=True,
                                           num_workers=16)




state_dict = torch.load("./save.pkl")
network.load_state_dict(state_dict)
dataiter = iter(test_loader)
network.eval()
def prediction(dataiter, network):


    batch = dataiter.next()
    inputs = Variable(batch['x'].cuda())
    origin = Variable(batch['origin'])[0]
    labels = Variable(batch['y'].cuda())[0]	
    output = network(inputs)[0]
    point = pcl.PointCloud_PointXYZRGBA()
    print batch['path']
    _point_list = []
    in_point = np.transpose(inputs[0].cpu().numpy(), (1, 0))
    print labels,output
    for i in range(len(in_point)):
        if labels[i].argmax() == 1:
            print output[i], labels[i]
        #if output[i].argmax() == labels[i] and output[i].argmax() == 1:
        #if output[i].argmax() == labels[i].argmax() and output[i].argmax() == 1:
        if output[i][0] < -0.1:
            red = float(origin[i][3] >> 16)
            green = float(origin[i][4] >> 8) 
            blue = float(origin[i][5] >> 0) 
            #print "1",red,green,blue      
            rgb = float(int(origin[i][5] << 16) | int(origin[i][4] << 8) | int(origin[i][3]))

            # # red = np.right_shift(red, 8).astype(np.uint8)
            # # green = np.right_shift(green, 8).astype(np.uint8)
            # # blue = np.right_shift(blue, 8).astype(np.uint8)
            # # print "2",red,green,blue      
            # red = np.asarray(red).astype(np.uint32)
            # green = np.asarray(green).astype(np.uint32)
            # blue = np.asarray(blue).astype(np.uint32)    
            # print "3",red,green,blue       
            # rgb = np.left_shift(red, 16) + np.left_shift(green, 8) + np.left_shift(blue, 0)

            #print "4",rgb        
            _point_list.append([inputs[0][0][i], inputs[0][1][i], inputs[0][2][i],rgb])

        # rgb = np.left_shift(red, 16) + np.left_shift(green, 8) + np.left_shift(blue, 0)
        # ptcloud = np.vstack((in_point[i][0], in_point[i][1], in_point[i][2], rgb)).transpose()

    point.from_array(np.asarray(_point_list,dtype = np.float32))
    point.to_file("./out.pcd") 

prediction(dataiter, network)
