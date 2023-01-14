import argparse
import os
import datetime
import random

import torch
import torch.optim as Optim

from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from dataset import ShapeNet
from models import PCN
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1, emd_loss
from visualization import plot_pcd_one_view


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, params.category)
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, params.category, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(params.exp_name), False)
    log(log_fd, "Logger directory: {}".format(logger_path), False)
    log(log_fd, str(params), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def train(params):
    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)

    log(log_fd, 'Loading Data...')

    train_dataset = ShapeNet('data/PCN', 'train', params.category)
    val_dataset = ShapeNet('data/PCN', 'valid', params.category)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    log(log_fd, "Dataset loaded!")

    # model
    model = PCN(num_dense=16384, latent_dim=1024, grid_size=4).to(params.device)

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999))
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    step = len(train_dataloader) // params.log_frequency

    # load pretrained model and optimizer
    if params.ckpt_path is not None:
        model.load_state_dict(torch.load(params.ckpt_path))

    # training
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    for epoch in range(1, params.epochs + 1):
        # hyperparameter alpha
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif train_step < 50000:
            alpha = 0.5
        else:
            alpha = 1.0

        # training
        model.train()
        for i, (p, c) in enumerate(train_dataloader):
            p, c = p.to(params.device), c.to(params.device)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(p)
            
            # loss function
            if params.coarse_loss == 'cd':
                loss1 = cd_loss_L1(coarse_pred, c)
            elif params.coarse_loss == 'emd':
                coarse_c = c[:, :1024, :]
                loss1 = emd_loss(coarse_pred, coarse_c)
            else:
                raise ValueError('Not implemented loss {}'.format(params.coarse_loss))
                
            loss2 = cd_loss_L1(dense_pred, c)
            loss = loss1 + alpha * loss2

            # back propagation
            loss.backward()
            optimizer.step()

            if (i + 1) % step == 0:
                log(log_fd, "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: coarse loss = {:.6f}, dense l1 cd = {:.6f}, total loss = {:.6f}"
                    .format(epoch, params.epochs, i + 1, len(train_dataloader), loss1.item() * 1e3, loss2.item() * 1e3, loss.item() * 1e3))
            
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
            train_step += 1
        
        lr_schedual.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization

            for i, (p, c) in enumerate(val_dataloader):
                p, c = p.to(params.device), c.to(params.device)
                coarse_pred, dense_pred = model(p)
                total_cd_l1 += l1_cd(dense_pred, c).item()

                # save into image
                if rand_iter == i:
                    index = random.randint(0, dense_pred.shape[0] - 1)
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                                      [p[index].detach().cpu().numpy(), coarse_pred[index].detach().cpu().numpy(), dense_pred[index].detach().cpu().numpy(), c[index].detach().cpu().numpy()],
                                      ['Input', 'Coarse', 'Dense', 'Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            
            total_cd_l1 /= len(val_dataset)
            val_writer.add_scalar('l1_cd', total_cd_l1, val_step)
            val_step += 1

            log(log_fd, "Validate Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, params.epochs, total_cd_l1 * 1e3))
        
        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_l1_cd.pth'))
            
    log(log_fd, 'Best l1 cd model in epoch {}, the minimum l1 cd is {}'.format(best_epoch_l1, best_cd_l1 * 1e3))
    log_fd.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCN')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    parser.add_argument('--coarse_loss', type=str, default='cd', help='loss function for coarse point cloud')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')
    params = parser.parse_args()
    
    train(params)
