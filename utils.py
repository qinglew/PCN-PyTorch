import random

import numpy as np
import torch
from open3d import *


def show_point_cloud(points: np.ndarray, rgd=None):
    assert points.ndim == 2

    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(points)
    if rgd is not None:
        point_cloud.paint_uniform_color(rgd)
    draw_geometries([point_cloud])


def setup_seed(seed):
    """
    Set the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
