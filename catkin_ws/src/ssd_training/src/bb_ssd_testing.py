#!/usr/bin/env python

import numpy as np
import cv2
import roslib
import rospy
import struct
import math
import time
import rospkg
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from ssd import build_ssd
import os 
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
import sys
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data

class bb_ssd_training(object):
	def __init__(self):
		r = rospkg.RosPack()
		self.image_pub = rospy.Publisher("/predict_img", Image, queue_size = 1)
		self.cv_bridge = CvBridge() 
		self.prob_threshold = 0.2
		self.labels = ['background' , # always index 0
				'bb_extinguisher','bb_drill','bb_backpack']
		path = r.get_path('ssd_training')
		self.root = subt_ROOT
		self.basenet = 'ssd300_subt_1000.pth'
		self.batch_size = 1
		self.num_workers = 10
		self.cuda = True
		self.start_iter = 0
		self.save_folder = "src/weights/"	

		if torch.cuda.is_available():
			if self.cuda:
				torch.set_default_tensor_type('torch.cuda.FloatTensor')
			if not self.cuda:
				print "WARNING: It looks like you have a CUDA device, but aren't " + \
				"using CUDA.\nRun with --cuda for optimal training speed."
				torch.set_default_tensor_type('torch.FloatTensor')	
		else:
			torch.set_default_tensor_type('torch.FloatTensor')
		self.cfg = subt
		self.dataset = subtDetection(root=self.root,image_sets=[('test')], transform=SSDAugmentation(self.cfg['min_dim'],MEANS))		
		self.ssd_net = build_ssd('test', self.cfg['min_dim'], self.cfg['num_classes'])
		if self.cuda:
			self.net = torch.nn.DataParallel(self.ssd_net)
			cudnn.benchmark = True

		weights = torch.load(os.path.join(path, self.save_folder, self.basenet))
		self.ssd_net.load_state_dict(weights)
		self.epoch_size = len(self.dataset) // self.batch_size
		self.net.eval()
		self.data_loader = data.DataLoader(self.dataset, self.batch_size,
							num_workers=self.num_workers,
							shuffle=True, collate_fn=detection_collate,
							pin_memory=True)
		self.batch_iterator = iter(self.data_loader)
		self.switch = False

		for iteration in range(self.start_iter, self.cfg['max_iter']):
			
			if self.switch:
				break;
			if iteration % self.epoch_size == 0:
				self.batch_iterator = iter(self.data_loader)

			# load train data
			images, rgb_img = next(self.batch_iterator)

			if self.cuda:
				images = Variable(images.cuda())


			# forward
			t0 = time.time()
			out = self.net(images)
			t1 = time.time()
			print (rgb_img[0]).cpu().numpy()
			rgb_img = (rgb_img[0]).cpu().numpy()
			rgb_img = rgb_img.astype(np.uint8)

			scale = torch.Tensor(rgb_img.shape[1::-1]).repeat(2)
			detections = out.data	# torch.Size([1, 4, 200, 5]) --> [batch?, class, object, coordinates]
			objs = []
			for i in range(detections.size(1)): # detections.size(1) --> class size
				for j in range(5):	# each class choose top 5 predictions
					if detections[0, i, j, 0].cpu().numpy() > self.prob_threshold:
						score = detections[0, i, j, 0]
						pt = (detections[0, i, j,1:]*scale).cpu().numpy()
						objs.append([pt[0], pt[1], pt[2]-pt[0]+1, pt[3]-pt[1]+1, i])
			print out
			for obj in objs:
				if obj[4] == 1:
					color = (0, 255, 255)
				elif obj[4] == 2:
					color = (255, 255, 0)
				elif obj[4] == 3:
					color = (255, 0, 255)
				else:
					color = (0, 0, 0)
				cv2.rectangle(rgb_img, (int(obj[0]), int(obj[1])),\
									(int(obj[0] + obj[2]), int(obj[1] + obj[3])), color, 3)
				cv2.putText(rgb_img, self.labels[obj[4]], (int(obj[0] + obj[2]), int(obj[1])), 0, 1, color,2)
			rgb_img = self.cv_bridge.cv2_to_imgmsg(rgb_img, "bgr8")
			self.image_pub.publish(rgb_img)


	def adjust_learning_rate(self, optimizer, gamma, step):
		lr = self.lr * (self.gamma ** (step))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def weights_init(self, m):
		if isinstance(m, nn.Conv2d):
			self.xavier(m.weight.data)
			m.bias.data.zero_()

	def xavier(self, param):
		init.xavier_uniform(param)

	def onShutdown(self):
		rospy.loginfo("Shutdown.")
		self.switch = True
		


if __name__ == '__main__': 
	rospy.init_node('bb_ssd_training',anonymous=False)
	bb_ssd_training = bb_ssd_training()
	rospy.on_shutdown(bb_ssd_training.onShutdown)
	rospy.spin()
