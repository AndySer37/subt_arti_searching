#!/usr/bin/env python

from InstanceSeg_net import *
from pointnet import *
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import rospy
import ctypes
from bb_pointnet.msg import *
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String,Float64, Bool, Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs import point_cloud2
import pcl
import time
import os
from bb_pointnet.srv import *

class bb_pointnet(object):
	def __init__(self):
		self.subt_CLASSES =  [  # always index 0 
				'background', 'bb_extinguisher', 'bb_drill']

		self.fx = 618.2425537109375
		self.fy = 618.5384521484375
		self.cx = 327.95947265625
		self.cy = 247.670654296875

		self.cv_bridge = CvBridge() 
		self.num_points = 10000
		
		#self.network = InstanceSeg(num_points = self.num_points)
		self.network = PointNetCls(k = 3) 
		self.network = self.network.cuda()
		model_dir = "/home/andyser/code/subt_related/subt_arti_searching/BB_for_pointnet/cls_weights"
		model_name = "pointnet_cls_epoch_95.pkl"	
		state_dict = torch.load(os.path.join(model_dir, model_name))
		self.network.load_state_dict(state_dict)

		self.predict_ser = rospy.Service("/pointnet_cls_prediction", pointnet_prediction, self.callback)

	def callback(self, req):
		self.network.eval()
		points_list = []
		color_list = []

		for data in point_cloud2.read_points(req.input_pc, skip_nans=True):
			points_list.append([data[0], data[1], data[2]])		
			color_list.append(data[3])

		point = np.asarray(points_list)
		color_list = np.asarray(color_list)
		if point.shape[0] < self.num_points:
			row_idx = np.random.choice(point.shape[0], self.num_points, replace=True)
		else:
			row_idx = np.random.choice(point.shape[0], self.num_points, replace=False)	

		point_in = torch.from_numpy(point[row_idx])  	## need to revise
		color_list = color_list[row_idx]

		point_in = np.transpose(point_in, (1, 0))
		point_in = point_in[np.newaxis,:]

		inputs = Variable(point_in).type(torch.FloatTensor).cuda()
		output = self.network(inputs)[0][0]

		print "-------------", self.subt_CLASSES[output.argmax()], output.argmax()
		print "-------------", output


		return "Finish"

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	


if __name__ == '__main__': 
	rospy.init_node('bb_pointnet_prediction',anonymous=False)
	bb_pointnet = bb_pointnet()
	rospy.on_shutdown(bb_pointnet.onShutdown)

	rospy.spin()