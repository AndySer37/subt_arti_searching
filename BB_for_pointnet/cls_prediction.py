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
num_points = 8000
network = PointNetCls(k = 3)  #(num_points = num_points)
#network = InstanceSeg(num_points = num_points)
network = network.cuda()

test_dataset = clsSeg_Dataset(data_path="/home/andyser/data/subt_real",num_point = num_points,image_sets = ['test'])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=1, shuffle=True,
                                           num_workers=16)


state_dict = torch.load("./cls_weights/pointnet_cls_epoch_28.pkl")
network.load_state_dict(state_dict)
dataiter = iter(test_loader)
network.eval()
def prediction(dataiter, network):


    batch = dataiter.next()
    inputs = Variable(batch['x'].cuda())
    origin = Variable(batch['origin'])[0]
    labels = Variable(batch['y'].cuda())[0]	
    output = network(inputs)[0][0]
    point_origin = pcl.PointCloud_PointXYZRGBA()
    _origin_list = []
    in_point = np.transpose(inputs[0].cpu().numpy(), (1, 0))
    print batch['id']
    print output.argmax()
    print output
    if output.argmax() == labels:
        print("right")
    else:
        print("wrong")
    for i in range(len(in_point)):
        red = float(origin[i][3] >> 16)
        green = float(origin[i][4] >> 8) 
        blue = float(origin[i][5] >> 0) 
        rgb = float(int(origin[i][5] << 16) | int(origin[i][4] << 8) | int(origin[i][3]))    
        _origin_list.append([inputs[0][0][i], inputs[0][1][i], inputs[0][2][i],rgb])



    point_origin.from_array(np.asarray(_origin_list,dtype = np.float32))
    point_origin.to_file("./origin.pcd")

prediction(dataiter, network)


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
        # rgb = np.left_shift(red, 16) + np.left_shift(green, 8) + np.left_shift(blue, 0)
        # ptcloud = np.vstack((in_point[i][0], in_point[i][1], in_point[i][2], rgb)).transpose()
