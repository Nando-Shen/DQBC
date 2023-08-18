from models import make_model, model_profile
from utils.config import make_config
import torch
import argparse
from datas.utils import imread_rgb
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite
import numpy as np
import os



def read(idx):
    # transforms = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    input_dir = '/home/curry/jshe2377/sim_keras_init_1'
    # result_dir = '/home/curry/jshe2377/csi_vis_result'
    data_list = []
    img0 = os.path.join(input_dir, '{}.jpg'.format(idx))
    img1 = os.path.join(input_dir, '{}.jpg'.format(idx+2))

    gt = os.path.join(input_dir, '{}.jpg'.format(idx+1))
    data_list.extend([img0, img1, gt])
    images = [Image.open(pth) for pth in data_list]
    size = (384, 192)
    images = [torch.tensor(img_.resize(size)).unsqueeze(0) for img_ in images]

    return images

def test(images, idx, model,device):
    # input_dir = '/home/curry/jshe2377/sim_keras_init_1'
    result_dir = '/home/curry/jshe2377/csi_vis_result'
    print('Evaluating for {}'.format(idx))
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():

        images = [img_.to(device) for img_ in images]
        out = model(images[0],images[1])['final']
        #     pred = pred[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)*255
        #     Image.fromarray(np.uint8(pred)).save(os.path.join(args.output_dir,'interp.png'))

        out = out[0].cpu().clamp(0.0, 1.0).numpy() * 255
        out = torch.tensor(out).unsqueeze(0)
        # print(images[0].size())
        # print(out.size())

        imwrite(images[0], result_dir + '/{}csi.jpg'.format(idx))
        imwrite(images[1], result_dir + '/{}csi.jpg'.format(idx+2))
        imwrite(out, result_dir + '/{}csi.jpg'.format(idx+1))
    return

# parser = argparse.ArgumentParser()
# parser.add_argument('--config',help='.yaml config file path')
# parser.add_argument('--gpu_id',type=int,default=0)
# parser.add_argument('--im0',type=str)
# parser.add_argument('--im1',type=str)
# parser.add_argument('--output_dir',type=str)
# args = parser.parse_args()
# cfg_file = args.config
# dev_id = args.gpu_id
# torch.cuda.set_device(dev_id)
#
# cfg = make_config(cfg_file, launch_experiment=False)
#
# print(model_profile(cfg.model))
#
# model = make_model(cfg.model)
# model.cuda()
# model.eval()

def evalvis(model):
    model.eval()
    device = torch.device('cuda')
    for i in range(1, 199, 2):
        images = read(i)
        test(images,i,model,device)

    # with torch.no_grad():
    #     im0 = TF.to_tensor(imread_rgb(args.im0))[None].cuda()
    #     im1 = TF.to_tensor(imread_rgb(args.im1))[None].cuda()
    #     pred = model(im0,im1)['final']
    #     pred = pred[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)*255
    #     Image.fromarray(np.uint8(pred)).save(os.path.join(args.output_dir,'interp.png'))

    
    
        
    