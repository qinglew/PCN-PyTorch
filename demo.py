import random
import torch

from open3d import *

from dataset.dataset import ShapeNet
from model import AutoEncoder
from loss import ChamferDistance
from utils import show_point_cloud


if __name__ == '__main__':
    cd_loss = ChamferDistance()

    test_dataset = ShapeNet(partial_path='/home/rico/Workspace/Dataset/shapenetpcn/partial',
                            gt_path='/home/rico/Workspace/Dataset/shapenetpcn/gt', split='test')

    network = AutoEncoder()
    network.load_state_dict(torch.load('log/epoch25_lr1e-3_alpha0.5.pth'))
    network = network.eval()

    partial_input, _, dense_gt = test_dataset[random.randint(0, len(test_dataset))]  # (2048, 3), (1024, 3), (16384, 3)
    # partial input
    show_point_cloud(partial_input.numpy())
    print("partial input point cloud has {} points".format(len(partial_input)))
    # dense ground truth
    show_point_cloud(dense_gt.numpy())
    print("dense output point cloud has {} points".format(len(dense_gt)))

    # prediction
    input_tensor = partial_input.unsqueeze(0).permute(0, 2, 1)
    _, _, output_tensor = network(input_tensor)
    
    temp1 = output_tensor.permute(0, 2, 1)
    temp2 = dense_gt.unsqueeze(0)
    loss = cd_loss(temp1, temp2)
    print('loss is {}'.format(loss))

    dense_pred = output_tensor.squeeze(0).permute(1, 0).cpu().detach().numpy()
    show_point_cloud(dense_pred)
    print("reconstructed point cloud has {} points".format(len(dense_pred)))
    print(partial_input.numpy())
    print()
    print(dense_pred)
