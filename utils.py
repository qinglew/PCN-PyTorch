import random

import torch
import numpy as np
from open3d import *


def show_point_cloud(points: np.ndarray, rgd=None):
    assert points.ndim == 2

    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(points)
    if rgd is not None:
        point_cloud.paint_uniform_color(rgd)
    draw_geometries([point_cloud])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


if __name__ == '__main__':
    pc = read_point_cloud('examples/chair.pcd')
    draw_geometries([pc])
    
    # resample pcd
    pc.points = Vector3dVector(resample_pcd(np.asarray(pc.points), 16384))
    print(len(pc.points))
    draw_geometries([pc])

    # for i in range(8):
    #     pc = read_point_cloud('examples/{}.pcd'.format(i))
    #     draw_geometries([pc])
