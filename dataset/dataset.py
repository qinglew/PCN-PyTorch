import os
import sys

import numpy as np
import torch
import torch.utils.data as Data


if __name__ == '__main__':
    DATASET_PATH = '/media/rico/BACKUP/Dataset/ShapeNetForPCN'
    for filename in os.listdir(DATASET_PATH):
        filename_path = os.path.join(DATASET_PATH, filename)
        if os.path.isdir(filename_path):
            print(filename + ": ", end='')
            print(len(os.listdir(filename_path)))
