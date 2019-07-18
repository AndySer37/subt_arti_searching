#!/usr/bin/env python

import numpy as np
import cv2
import roslib
import rospy
import struct
import math
import time
import rospkg
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage
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
		path = r.get_path('ssd_training')

		self.root = subt_ROOT
		self.basenet = 'vgg16_reducedfc.pth'
		self.batch_size = 16
		self.start_iter = 0
		self.num_workers = 10
		self.cuda = True
		self.lr = 1e-4
		self.momentum = 0.9
		self.weight_decay = 5e-4
		self.gamma = 0.1
		self.save_folder = "src/weights/"	
		self.resume = None	

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
		self.dataset = subtDetection(root=self.root,transform=SSDAugmentation(self.cfg['min_dim'],MEANS))		
		self.ssd_net = build_ssd('train', self.cfg['min_dim'], self.cfg['num_classes'])
		if self.cuda:
			self.net = torch.nn.DataParallel(self.ssd_net)
			cudnn.benchmark = True
		# if self.resume:
		# 	print 'Resuming training, loading {}...'.format(self.resume)
		# 	self.ssd_net.load_weights(self.resume)
		# else:
		# 	vgg_weights = torch.load(os.path.join(path, self.save_folder, self.basenet))
		# 	print 'Loading base network...'
		# 	self.ssd_net.vgg.load_state_dict(vgg_weights)

		if not self.resume:
			print 'Initializing weights...'
			# initialize newly added layers' weights with xavier method
			self.ssd_net.extras.apply(self.weights_init)
			self.ssd_net.loc.apply(self.weights_init)
			self.ssd_net.conf.apply(self.weights_init)

		self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum,weight_decay=self.weight_decay)
		self.criterion = MultiBoxLoss(self.batch_size ,self.cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, self.cuda)
		self.net.train()
		self.loc_loss = 0
		self.conf_loss = 0
		self.epoch = 0		
		print 'Loading the dataset...'
		self.epoch_size = len(self.dataset) // self.batch_size
		self.step_index = 0	
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
				# reset epoch loss counters
				self.loc_loss = 0
				self.conf_loss = 0
				self.batch_iterator = iter(self.data_loader)
				self.epoch += 1

			if iteration in self.cfg['lr_steps']:
				self.step_index += 1
				self.adjust_learning_rate(self.optimizer, self.gamma, self.step_index)

			# load train data
			images, targets = next(self.batch_iterator)

			if self.cuda:
				images = Variable(images.cuda())
				targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
			else:
				images = Variable(images)
				targets = [Variable(ann, volatile=True) for ann in targets]
			# forward
			t0 = time.time()
			out = self.net(images)
			# backprop
			self.optimizer.zero_grad()
			loss_l, loss_c = self.criterion(out, targets)
			loss = loss_l + loss_c
			loss.backward()
			self.optimizer.step()
			t1 = time.time()
			self.loc_loss += loss_l.data
			self.conf_loss += loss_c.data

			if iteration % 10 == 0:
				print 'timer: %.4f sec.' % (t1 - t0)
				print 'iter ' + repr(iteration) ,loss.item()


			if iteration != 0 and iteration % 1000 == 0:
				print 'Saving state, iter:', iteration
				torch.save(self.ssd_net.state_dict(), os.path.join(path, "src/weights","ssd300_subt_" + repr(iteration) + '.pth'))

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
