import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from losses import make_loss
import os.path as osp

from utils.config import make_config
from benchmark.utils.pytorch_msssim import ssim_matlab

from dataset import VimeoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from test import evalvis

# from config import *
from models import make_model, model_profile


device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]


def make_optimizer(cfg, model):
    """ Create the optimizer and learning rate scheduler """

    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=cfg.wdecay)

    return optimizer


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5

def train(model, local_rank, batch_size, data_path, cfg):
    if local_rank == 0:
        writer = SummaryWriter('log/train_EMAVFI')
    optimizer = make_optimizer(cfg.train, model)
    save_path = osp.join(cfg.ckp_root, '%s.pth' % (cfg.exp_name))

    step = 0
    nr_eval = 0
    best = 0
    dataset = VimeoDataset('train', data_path)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('test', data_path)
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    loss_fn = make_loss(cfg)
    for epoch in range(300):
        sampler.set_epoch(epoch)
        for i, imgs in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step)
            pred = model(imgs[:, 0:3], imgs[:, 3:6])
            loss, metrics = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss))
            step += 1
        nr_eval += 1
        if nr_eval % 0 == 0:
            # evaluate(model, val_data, nr_eval, local_rank)
            evalvis(model)

        torch.save(model.state_dict(), save_path)
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank):
    if local_rank == 0:
        writer_val = SummaryWriter('log/validate_EMAVFI')
    path = '/home/curry/jshe2377/atd12k_points'
    f = os.listdir(os.path.join(path, 'test_2k_540p'))
    psnr_list, ssim_list = [], []

    for i in f:
        name = str(i).strip()
        size = (384, 192)
        if (len(name) <= 1):
            continue
        I0 = cv2.imread(path + '/test_2k_540p/' + name + '/frame1.jpg')
        I1 = cv2.imread(path + '/test_2k_540p/' + name + '/frame2.jpg')
        I2 = cv2.imread(path + '/test_2k_540p/' + name + '/frame3.jpg')  # BGR -> RBG
        I0 = cv2.resize(I0, size)
        I1 = cv2.resize(I1, size)
        I2 = cv2.resize(I2, size)
        I0 = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
        I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
        with torch.no_grad():
            mid = model(I0, I2)['final'][0]
        ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).cuda().unsqueeze(0) / 255.,
                           mid.unsqueeze(0)).detach().cpu().numpy()
        mid = mid.detach().cpu().numpy().transpose(1, 2, 0)
        I1 = I1 / 255.
        psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
        # os.makedirs('/home/curry/jshe2377/dqtest/' + name)
        mid = mid * 255.
        cv2.imwrite(r"/home/curry/jshe2377/dqtest/" + name + "/emavfi.jpg", mid)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))

    #
    # psnr = []
    # for _, imgs in enumerate(val_data):
    #     imgs = imgs.to(device, non_blocking=True) / 255.
    #     imgs, gt = imgs[:, 0:6], imgs[:, 6:]
    #     with torch.no_grad():
    #         pred = model(imgs[:, 0:3], imgs[:, 3:6])['final']
    #     for j in range(gt.shape[0]):
    #         psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
    #
    # psnr = np.array(psnr).mean()
    # if local_rank == 0:
    #     print(str(nr_eval), psnr)
    #     writer_val.add_scalar('psnr', psnr, nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--data_path', type=str, help='data path of vimeo90k')
    parser.add_argument('--config', type=str, help='config')

    args = parser.parse_args()

    cfg = make_config(cfg_file=args.config)
    model = make_model(cfg.model)

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # model = Model(args.local_rank)
    train(model, args.local_rank, args.batch_size, args.data_path, cfg)
        
