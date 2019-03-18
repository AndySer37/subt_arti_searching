#!/usr/bin/env python

from InstanceSeg_net import *
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import rospy

from bb_pointnet.msg import *
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String,Float64, Bool, Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs import point_cloud2
import time
import os


class bb_pointnet(object):
	def __init__(self):
		self.subt_CLASSES =  [  # always index 0
				'bb_extinguisher']

		self.fx = 618.2425537109375
		self.fy = 618.5384521484375
		self.cx = 327.95947265625
		self.cy = 247.670654296875

		self.cv_bridge = CvBridge() 
		self.num_points = 8000
		self.network = InstanceSeg(num_points = self.num_points)
		self.network = self.network.cuda()
		model_dir = "/home/andyser/code/subt_related/subt_arti_searching/BB_for_pointnet/weights"
		model_name = "pointnet_epoch_55.pkl"	
		state_dict = torch.load(os.path.join(model_dir, model_name))
		self.network.load_state_dict(state_dict)
		self.prediction = rospy.Publisher('/prediction', PointCloud2, queue_size=10)
		self.origin = rospy.Publisher('/origin', PointCloud2, queue_size=10)

		rospy.Subscriber("/input", bb_input, self.callback)

	def callback(self, msg):

		# img = cv2.imread("/home/andyser/data/subt_real/image/extinguisher/scene000001/1.jpg",cv2.IMREAD_UNCHANGED)
		# depth = cv2.imread("/home/andyser/data/subt_real/depth/extinguisher/scene000001/1.png",cv2.IMREAD_UNCHANGED)
		# mask = cv2.imread("/home/andyser/data/subt_real/mask/extinguisher/scene000001/1.png",cv2.IMREAD_UNCHANGED)

		try:
			img = self.cv_bridge.imgmsg_to_cv2(msg.image, "bgr8")
			depth = self.cv_bridge.imgmsg_to_cv2(msg.depth, "16FC1")
			mask = self.cv_bridge.imgmsg_to_cv2(msg.mask, "64FC1")
		except CvBridgeError as e:
			print(e)

		(h, w, c) = img.shape		
		point = list()
		origin = list()
		for i in range(h):
			for j in range(w):
				if mask[i,j] == 255:
					z = depth[i,j]
					if z > 1:
						x, y, z = self.getXYZ(j,i,z/1000.)
						point.append([z,-y,-x])
						(r,g,b) = img[i,j]
						origin.append([z,-y,-x,r,g,b])
						
		point = np.asarray(point, dtype = np.float32)
		origin = np.asarray(origin, dtype = np.float32)	

		if point.shape[0] < self.num_points:
			row_idx = np.random.choice(point.shape[0], self.num_points, replace=True)
		else:
			row_idx = np.random.choice(point.shape[0], self.num_points, replace=False)	

		point_in = torch.from_numpy(point[row_idx,:3])  	## need to revise
		origin = origin[row_idx]
		target = torch.zeros((self.num_points,2))
		point_in = np.transpose(point_in, (1, 0))

		point_in = point_in[np.newaxis,:]
		inputs = Variable(point_in.cuda())
		output = self.network(inputs)[0]

		_point_list = []
		_origin_list = []
		a = 255
		for i in range(self.num_points):
			rgb = struct.unpack('I', struct.pack('BBBB', int(origin[i][3]), int(origin[i][4]), int(origin[i][5]), a))[0] 
			#if output[i].argmax() == labels[i] and output[i].argmax() == 1:
			if output[i].argmax() == 1:
			#if output[i][0] < -0.1:
				_point_list.append([inputs[0][0][i], inputs[0][1][i], inputs[0][2][i],rgb])
			_origin_list.append([inputs[0][0][i], inputs[0][1][i], inputs[0][2][i],rgb])

		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id ="camera_link"

		fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1), PointField('z', 8, PointField.FLOAT32, 1), PointField('rgb', 12, PointField.UINT32, 1)]
		pointcloud_pre = point_cloud2.create_cloud(header, fields, _point_list)
		pointcloud_origin = point_cloud2.create_cloud(header, fields, _point_list)

		self.prediction.publish(pointcloud_pre)
		self.origin.publish(pointcloud_pre)

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	


	def getXYZ(self,xp, yp, zc):
		#### Definition:
		# cx, cy : image center(pixel)
		# fx, fy : focal length
		# xp, yp: index of the depth image
		# zc: depth
		inv_fx = 1.0/self.fx
		inv_fy = 1.0/self.fy
		x = (xp-self.cx) *  zc * inv_fx
		y = (yp-self.cy) *  zc * inv_fy
		z = zc
		return (x,y,z)			


if __name__ == '__main__': 
	rospy.init_node('bb_pointnet_prediction',anonymous=False)
	bb_pointnet = bb_pointnet()
	rospy.on_shutdown(bb_pointnet.onShutdown)

	# while(1):
	# 	bb_pointnet.callback()

	rospy.spin()