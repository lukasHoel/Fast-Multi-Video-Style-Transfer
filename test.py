import argparse
import os
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from network import net
from network import styler
from torchvision.utils import save_image

#import mmcv
import time
import logging
from torch.autograd import Variable



# cudnn.benchmark = True
# Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=0.0)
parser.add_argument('--rec_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--temporal_weight', type=float, default=1.0)
parser.add_argument('--total_weight', type=float, default=2e-8)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--scene_dir', type=str, required=True)

def train_transform():
    transform_list = [
        # transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def style_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


if __name__ == '__main__':

    args = parser.parse_args()
    styler = styler.ReCoNet()
    styler.eval()
    styler.load_state_dict(torch.load('21style_nolstm_noflow.pth.tar', map_location="cpu")['state_dict'], strict=False)
    network = net.Net(styler, vgg=None)
    network = network.cuda()
    network.styler = network.styler.cuda()

    if args.parallel:
        network = nn.DataParallel(network)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    content_tf = train_transform()
    style_tf = train_transform()

    print('loading dataset done', flush=True)

    images = [f for f in os.listdir(args.scene_dir) if 'jpg' in f or 'png' in f]
    images = sorted(images, key=lambda x: int(x.split(".")[0]))
    for bank in range(21):
        for i, f in enumerate(images):
            print(i, bank, flush=True)
            cimg = Image.open(os.path.join(args.scene_dir, f)).convert('RGB')
            cimg = content_tf(cimg).unsqueeze(0).cuda()
            out = network.evaluate(cimg, bank)
            if not os.path.exists(f"{args.save_dir}/{bank}"):
                os.mkdir(f"{args.save_dir}/{bank}")

            save_image(out, f"{args.save_dir}/{bank}/{i}.jpg")

    #mmcv.frames2video('output', 'mst_cat_flow.avi', fps=6)


