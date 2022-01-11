import os
import argparse

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data

from models import PCN
from dataset import ShapeNet
from visualization import plot_pcd_one_view
from metrics.metric import l1_cd, l2_cd, emd, f_score


CATEGORIES_PCN       = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
CATEGORIES_PCN_NOVEL = ['bus', 'bed', 'bookshelf', 'bench', 'guitar', 'motorbike', 'skateboard', 'pistol']


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def test_single_category(category, model, params, save=True):
    if save:
        cat_dir = os.path.join(params.result_dir, category)
        image_dir = os.path.join(cat_dir, 'image')
        output_dir = os.path.join(cat_dir, 'output')
        make_dir(cat_dir)
        make_dir(image_dir)
        make_dir(output_dir)

    test_dataset = ShapeNet('/media/server/new/datasets/PCN', 'test_novel' if params.novel else 'test', category)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    index = 1
    total_l1_cd, total_l2_cd, total_f_score = 0.0, 0.0, 0.0
    with torch.no_grad():
        for p, c in test_dataloader:
            p = p.to(params.device)
            c = c.to(params.device)
            _, c_ = model(p)
            total_l1_cd += l1_cd(c_, c).item()
            total_l2_cd += l2_cd(c_, c).item()
            for i in range(len(c)):
                input_pc = p[i].detach().cpu().numpy()
                output_pc = c_[i].detach().cpu().numpy()
                gt_pc = c[i].detach().cpu().numpy()
                total_f_score += f_score(output_pc, gt_pc)
                if save:
                    plot_pcd_one_view(os.path.join(image_dir, '{:03d}.png'.format(index)), [input_pc, output_pc, gt_pc], ['Input', 'Output', 'GT'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                    export_ply(os.path.join(output_dir, '{:03d}.ply'.format(index)), output_pc)
                index += 1
    
    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
    avg_f_score = total_f_score / len(test_dataset)

    return avg_l1_cd, avg_l2_cd, avg_f_score


def test(params, save=False):
    if save:
        make_dir(params.result_dir)

    print(params.exp_name)

    # load pretrained model
    model = PCN(16384, 1024, 4).to(params.device)
    model.load_state_dict(torch.load(params.ckpt_path))
    model.eval()

    print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)', 'FScore-0.01(%)'))
    print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------'))

    if params.category == 'all':
        if params.novel:
            categories = CATEGORIES_PCN_NOVEL
        else:
            categories = CATEGORIES_PCN
        
        l1_cds, l2_cds, fscores = list(), list(), list()
        for category in categories:
            avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(category, model, params, save)
            print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))
            l1_cds.append(avg_l1_cd)
            l2_cds.append(avg_l2_cd)
            fscores.append(avg_f_score)
        
        print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------'))
        print('\033[32m{:20s}{:<20.4f}{:<20.4f}{:<20.4f}\033[0m'.format('Average', np.mean(l1_cds) * 1e3, np.mean(l2_cds) * 1e4, np.mean(fscores) * 1e2))
    else:
        avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(params.category, model, params, save)
        print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(params.category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))


def test_single_category_emd(category, model, params):
    test_dataset = ShapeNet('/media/server/new/datasets/PCN', 'test_novel' if params.novel else 'test', category)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    total_emd = 0.0
    with torch.no_grad():
        for p, c in test_dataloader:
            p = p.to(params.device)
            c = c.to(params.device)
            _, c_ = model(p)
            total_emd += emd(c_, c).item()
        
    avg_emd = total_emd / len(test_dataset) / c_.shape[1]
    return avg_emd


def test_emd(params):
    print(params.exp_name)

    # load pretrained model
    model = PCN(16384, 1024, 4).to(params.device)
    model.load_state_dict(torch.load(params.ckpt_path))
    model.eval()

    print('\033[33m{:20s}{:20s}\033[0m'.format('Category', 'EMD(1e-3)'))
    print('\033[33m{:20s}{:20s}\033[0m'.format('--------', '---------'))

    if params.category == 'all':
        if params.novel:
            categories = CATEGORIES_PCN_NOVEL
        else:
            categories = CATEGORIES_PCN
        
        emds = list()
        for category in categories:
            avg_emd = test_single_category_emd(category, model, params)
            print('{:20s}{:<20.4f}'.format(category.title(), 1e3 * avg_emd))
            emds.append(avg_emd)
        
        print('\033[33m{:20s}{:20s}\033[0m'.format('--------', '---------'))
        print('\033[32m{:20s}{:<20.4f}\033[0m'.format('Average', np.mean(emds) * 1e3))
    else:
        avg_emd = test_single_category_emd(params.category, model, params)
        print('{:20s}{:<20.4f}'.format(params.category.title(), 1e3 * avg_emd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, help='The path of pretrained model.')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=6, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    parser.add_argument('--emd', type=bool, default=False, help='Whether evaluate emd')
    params = parser.parse_args()

    if not params.emd:
        test(params, params.save)
    else:
        test_emd(params)
