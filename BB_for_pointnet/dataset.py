import sys
import pcl
import torch
import torch.utils.data
import pandas as pd
import os.path as osp
import copy
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import math
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
subt_CLASSES =  [  # always index 0
    'bb_extinguisher']
HOME = osp.expanduser("~")
# note: if you used our download scripts, this should be right
subt_ROOT = osp.join(HOME, "data/subt_real/")


class subtAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(subt_CLASSES, range(len(subt_CLASSES))))
        self.keep_difficult = keep_difficult
    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            #difficult = int(obj.find('difficult').text) == 1
            #if not self.keep_difficult and difficult:
            #    continue
            name = obj.find('name').text.lower().strip()
            if name not in self.class_to_ind:
                continue
            bbox = obj.find('bndbox')
            if bbox is not None:
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    # scale height or width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                label_idx = self.class_to_ind[name]
                bndbox.append(label_idx)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            else: # For LabelMe tool
                polygons = obj.find('polygon')
                x = []
                y = []
                bndbox = []
                for polygon in polygons.iter('pt'):
                    # scale height or width
                    #x.append(int(polygon.find('x').text) / width)
                    #y.append(int(polygon.find('y').text) / height)
                    x.append(int(polygon.find('x').text))
                    y.append(int(polygon.find('y').text))               
                bndbox.append(min(x))
                bndbox.append(min(y))
                bndbox.append(max(x))
                bndbox.append(max(y))
                label_idx = self.class_to_ind[name]
                bndbox.append(label_idx)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class InstanceSeg_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_point,image_sets=['train'],
                 transform=None, target_transform=subtAnnotationTransform(), dataset_name='subt'):
        self.fx = 618.2425537109375
        self.fy = 618.5384521484375
        self.cx = 327.95947265625
        self.cy = 247.670654296875

        self.num_point = num_point
        self.root = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'image', '%s.jpg')
        self._maskpath = osp.join('%s', 'mask', '%s.png')
        self._depthpath = osp.join('%s', 'depth', '%s.png')
        self.ids = list()
        for name in image_sets:
            rootpath = osp.join(self.root)
            for line in open(osp.join(rootpath, 'ImageSets/Main', name + '.txt')):
                self.ids.append((rootpath, line.strip().split(' ')[0]))


    def __getitem__(self, index):
        
        im, mask, depth, gt, h, w, img_id = self.pull_item(index)
        lower_bound = -5
        upper_bound = 15
        
        gt = gt[0]
        x_bb = gt[3] - gt[1]
        y_bb = gt[2] - gt[0]
        _min = min(x_bb, y_bb)
        if _min < 5:
            lower_bound = 0
        bonus = np.random.random_integers(lower_bound, upper_bound, 1)
        #bonus = 5
        point = list()
        label = list()
        origin = list()

        # for i in range(len(gt)):
        #     gt[i] += 20

        for i in range(h):
            for j in range(w):
                if gt[0] - bonus <= j <= gt[2] + bonus and gt[1] - bonus <= i <= gt[3] + bonus:
                    z = depth[i,j]
                    if z > 1:
                        x, y, z = self.getXYZ(j,i,z/1000.)
                        point.append([z,-y,-x])
                        (r,g,b) = im[i,j]
                        origin.append([z,-y,-x,r,g,b])
                        label.append([mask[i,j]/255])
        point = np.asarray(point, dtype = np.float32)
        label = np.asarray(label, dtype = np.float32)
        origin = np.asarray(origin, dtype = np.float32)
        #print point.shape[0]

        if point.shape[0] < self.num_point:
            row_idx = np.random.choice(point.shape[0], self.num_point, replace=True)
        else:
            row_idx = np.random.choice(point.shape[0], self.num_point, replace=False)        


        point_out = torch.from_numpy(point[row_idx,:3])  	## need to revise
        origin = origin[row_idx]
        label = label[row_idx]			## need to revise
        target = torch.zeros((self.num_point,2))
        # #target = torch.zeros((self.num_point), dtype = torch.long)
        for i in range(self.num_point):
            #print label[i]
            #target[i] = int(label[i])
            target[i][int(label[i])] = 1
        point_out = np.transpose(point_out, (1, 0))

        out = {'x': point_out, 'y': target, 'id': img_id, 'origin': origin}
        return out


    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id,cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(self._depthpath % img_id,cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self._maskpath % img_id,cv2.IMREAD_UNCHANGED)
        height, width, channels = img.shape
        
        if self.target_transform is not None:
            ### targe = [[xmin, ymin, xmax, ymax, label_ind], ... ]
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, mask, depth, target, height, width, img_id

        # return torch.from_numpy(img).permute(2, 0, 1), target, height, width

        ########## return torch.from_numpy(img), target, height, width ############

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

class clsSeg_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_point,image_sets=['train'],
                 transform=None, target_transform=subtAnnotationTransform(), dataset_name='subt'):
        self.fx = 618.2425537109375
        self.fy = 618.5384521484375
        self.cx = 327.95947265625
        self.cy = 247.670654296875

        self.num_point = num_point
        self.root = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'image', '%s.jpg')
        self._maskpath = osp.join('%s', 'mask', '%s.png')
        self._depthpath = osp.join('%s', 'depth', '%s.png')
        self.ids = list()
        for name in image_sets:
            rootpath = osp.join(self.root)
            for line in open(osp.join(rootpath, 'ImageSets/Main', name + '.txt')):
                self.ids.append((rootpath, line.strip().split(' ')[0]))


    def __getitem__(self, index):
        
        im, mask, depth, gt, h, w, img_id = self.pull_item(index)
        lower_bound = 15
        upper_bound = 20
        
        gt = gt[0]
        x_bb = gt[3] - gt[1]
        y_bb = gt[2] - gt[0]
        _min = min(x_bb, y_bb)
        if _min < 5:
            lower_bound = 0
        bonus = np.random.random_integers(lower_bound, upper_bound, 1)
        #bonus = 5
        point = list()
        label = list()
        origin = list()

        # for i in range(len(gt)):
        #     gt[i] += 20
        _is = True
        loop_count = 0
        gt_copy = copy.copy(gt)
        
        if np.random.random_integers(0, 2, 1) == 1:
            while True:
                gt = copy.copy(gt_copy)
                loop_count += 1
                ram = np.random.random_integers(0, 1, 1)
                if loop_count > 10:
                    gt = copy.copy(gt_copy)
                    break
                if ram == 0:
                    gt[0] = gt[0] - y_bb - 20
                    gt[2] = gt[2] - y_bb - 10
                elif ram == 1:
                    gt[0] = gt[0] + y_bb + 10 
                    gt[2] = gt[2] + y_bb + 20

                if 600 > gt[2] > 50  and 600 > gt[0] > 50:
                    _is = False
                    break
        
        for i in range(h):
            for j in range(w):
                if gt[0] - bonus <= j <= gt[2] + bonus and gt[1] - bonus <= i <= gt[3] + bonus:
                    z = depth[i,j]
                    if z > 1:
                        x, y, z = self.getXYZ(j,i,z/1000.)
                        point.append([z,-y,-x])
                        (r,g,b) = im[i,j]
                        origin.append([z,-y,-x,r,g,b])
        if _is:
            for i in range(len(subt_CLASSES)):
                if subt_CLASSES[i][3:] in self.ids[index][1]:
                    label.append(i+1)
        else:
            label.append(0)

        point = np.asarray(point, dtype = np.float32)
        label = np.asarray(label, dtype = np.float32)
        origin = np.asarray(origin, dtype = np.float32)
        ##print point.shape[0]
        if point.shape[0]==0:
            print gt,self.ids[index][1],gt_copy

        if point.shape[0] < self.num_point:
            row_idx = np.random.choice(point.shape[0], self.num_point, replace=True)
        else:
            row_idx = np.random.choice(point.shape[0], self.num_point, replace=False)        


        point_out = torch.from_numpy(point[row_idx,:3])     ## need to revise
        origin = origin[row_idx]

        target = torch.from_numpy(label).type(torch.LongTensor)
        # #target = torch.zeros((self.num_point), dtype = torch.long)

        point_out = np.transpose(point_out, (1, 0))

        out = {'x': point_out, 'y': target, 'id': img_id, 'origin': origin}
        return out


    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id,cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(self._depthpath % img_id,cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self._maskpath % img_id,cv2.IMREAD_UNCHANGED)
        height, width, channels = img.shape
        
        if self.target_transform is not None:
            ### targe = [[xmin, ymin, xmax, ymax, label_ind], ... ]
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, mask, depth, target, height, width, img_id

        # return torch.from_numpy(img).permute(2, 0, 1), target, height, width

        ########## return torch.from_numpy(img), target, height, width ############

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