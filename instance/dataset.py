import sys
import pcl
import torch
import torch.utils.data
import pandas as pd
import os
import pickle
import numpy as np
import math

subt_CLASSES = ('extinguisher','backpack','radio')



class InstanceSeg_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type, num_point,dataset_name='subt'):
        self.num_point = num_point
        self.root = data_path
        self.img_dir = self.root + "/image/"
        self.point_dir = self.root + "/pc/"
        self.label = self.root + "/" + type + ".csv"
        self.data  = pd.read_csv(self.label)

    def __getitem__(self, index):
        pcd   = self.data.iloc[index, 0]
        pcd_origin   = self.data.iloc[index, 1]
        point = pcl.PointCloud_PointXYZRGBA()
        point_origin = pcl.PointCloud_PointXYZRGBA()
        point.from_file(pcd)
        point_origin.from_file(pcd_origin)
        point_np = np.zeros((point.size,4), np.float32)
        point_origin_np = np.zeros((point.size,6), np.float32)
        for i in range(point.size):
            point_np[i][0] = point[i][0]
            point_np[i][1] = point[i][1]
            point_np[i][2] = point[i][2]
            point_np[i][3] = int(point[i][3]) >> 16 & 0x000ff

        if point_np.shape[0] < self.num_point:
            row_idx = np.random.choice(point_np.shape[0], self.num_point, replace=True)
        else:
            row_idx = np.random.choice(point_np.shape[0], self.num_point, replace=False)
        row_idx = np.sort(row_idx)
        point_out = torch.from_numpy(point_np[row_idx,:3])  	## need to revise
        label = torch.from_numpy(point_np[row_idx,3])			## need to revise
        
        origin_list = []
        for i in range(point.size):
            point_origin_np[i][0] = point_origin[i][0]
            point_origin_np[i][1] = point_origin[i][1]
            point_origin_np[i][2] = point_origin[i][2]
            point_origin_np[i][3] = int(point_origin[i][3]) >> 0 & 0x000ff
            point_origin_np[i][4] = int(point_origin[i][3]) >> 8 & 0x000ff
            point_origin_np[i][5] = int(point_origin[i][3]) >> 16 & 0x000ff

        #print point_origin_np[0] , point_origin_np[1]

        point_origin_np = point_origin_np[row_idx]
        # print origin_list[0] , origin_list[1]


        point_out = np.transpose(point_out, (1, 0))

        target = torch.zeros((self.num_point,2))
        #target = torch.zeros((self.num_point), dtype = torch.long)
        for i in range(self.num_point):
        	#print label[i]
        	#target[i] = int(label[i])
        	target[i][int(label[i])] = 1
        out = {'x': point_out, 'y': target, 'origin': point_origin_np ,'path': pcd_origin}
        return out


    def __len__(self):
        return len(self.data)

    # def pull_item(self, index):
    #     img_id = self.ids[index]

    #     target = ET.parse(self._annopath % img_id).getroot()
    #     img = cv2.imread(self._imgpath % img_id)
    #     height, width, channels = img.shape

    #     if self.target_transform is not None:
    #         target = self.target_transform(target, width, height)

    #     if self.transform is not None:
    #         target = np.array(target)
    #         img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
    #         # to rgb
    #         img = img[:, :, (2, 1, 0)]
    #         # img = img.transpose(2, 0, 1)
    #         target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
    #     return torch.from_numpy(img).permute(2, 0, 1), target, height, width