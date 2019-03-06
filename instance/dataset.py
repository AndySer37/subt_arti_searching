import sys

import torch
import torch.utils.data

import os
import pickle
import numpy as np
import math

subt_CLASSES = ('toolbox','backpack','radio')

class Transform(object):

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(subt_CLASSES, range(len(subt_CLASSES))))
        self.keep_difficult = keep_difficult
    def __call__(self, target, width, height):

        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

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
            # img_id = target.find('filename').text[:-4]
        #print(res)
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class InstanceSeg_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type,image_sets=[('2007', 'train')],
                 transform=None, target_transform=subtAnnotationTransform(),
                 dataset_name='subt'):
        
        self.root = data_path
        self.img_dir = self.root + "/image/"
        self.point_dir = self.root + "/point/"
        self.label = self.root + "/" + type + ".txt"

        
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'subt')
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        self.num_examples = len(self.ids)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

    def __len__(self):
        return self.num_examples

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width