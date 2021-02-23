import os
import random

import numpy as np
import torch
from torch._C import dtype
import torch.utils.data as Data
from open3d import *

import sys
sys.path.append('.')
sys.path.append('..')
from utils import resample_pcd, show_point_cloud


class ShapeNet(Data.Dataset):
    def __init__(self, partial_path, gt_path, num_input=2048, num_coarse=1024, num_dense=16384, split='train', num_scans=8):
        self.partial_path = partial_path
        self.gt_path = gt_path
        self.num_input = num_input
        self.num_coarse = num_coarse
        self.num_dense = num_dense
        
        with open('dataset/car_split/{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]
        
        self.metadata = list()
        for filename in filenames:
            for i in range(num_scans):
                partial_input = os.path.join(partial_path, 'pcd', filename, '{}.pcd'.format(i))
                ground_truth = os.path.join(gt_path, filename, 'model.pcd')
                self.metadata.append((partial_input, ground_truth))

    def __getitem__(self, index):
        partial_input_path, gt_output_path = self.metadata[index]
        partial_input = np.asarray(read_point_cloud(partial_input_path).points, dtype='f4')
        gt_output = np.asarray(read_point_cloud(gt_output_path).points, dtype='f4')

        partial_input = resample_pcd(partial_input, self.num_input)
        choice = np.random.choice(len(gt_output), self.num_coarse, replace=True)
        coarse_gt = gt_output[choice, :]
        dense_gt = resample_pcd(gt_output, self.num_dense)

        # to torch tensor
        partial_input = torch.from_numpy(partial_input)
        coarse_gt = torch.from_numpy(coarse_gt)
        dense_gt = torch.from_numpy(dense_gt)

        return partial_input, coarse_gt, dense_gt

    def __len__(self):
        return len(self.metadata)


if __name__ == '__main__':
    ROOT = "/home/rico/Workspace/Dataset/shapenetpcn"
    GT_ROOT = os.path.join(ROOT, 'gt')
    PARTIAL_ROOT = os.path.join(ROOT, 'partial')

    train_dataset = ShapeNet(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='train')
    val_dataset = ShapeNet(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='val')
    test_dataset = ShapeNet(partial_path=PARTIAL_ROOT, gt_path=GT_ROOT, split='test')
    print("\033[33mTraining dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(train_dataset)))
    print("\033[33mValidation dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(val_dataset)))
    print("\033[33mTesting dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(test_dataset)))

    # visualization
    input_pc, coarse_pc, dense_pc = train_dataset[random.randint(0, len(train_dataset))]
    show_point_cloud(input_pc.numpy())
    print("partial input point cloud has {} points".format(len(input_pc)))
    show_point_cloud(coarse_pc.numpy())
    print("coarse output point cloud has {} points".format(len(coarse_pc)))
    show_point_cloud(dense_pc.numpy())
    print("dense output point cloud has {} points".format(len(dense_pc)))
