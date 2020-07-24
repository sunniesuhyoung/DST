
# https://github.com/sunniesuhyoung/DST

# Various helper functions

import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# Image loading functions from https://github.com/futscdav/strotss
################################################################################
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), Image.BICUBIC)
    return resized

def pil_to_tensor(pil):
    return (torch.Tensor(np.array(pil).astype(np.float) / 127.5) - 1.0).permute((2,0,1)).unsqueeze(0)

################################################################################
# Feature sampling functions https://github.com/futscdav/strotss
################################################################################
def sample_indices(feat_content, feat_style):
    indices = None
    const = 128**2 # 32k or so
    feat_dims = feat_style.shape[1]
    big_size = feat_content.shape[2] * feat_content.shape[3]

    stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
    offset_x = np.random.randint(stride_x)
    stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
    offset_y = np.random.randint(stride_y)
    xx, xy = np.meshgrid(np.arange(feat_content.shape[2])[offset_x::stride_x],
                            np.arange(feat_content.shape[3])[offset_y::stride_y])

    xx = xx.flatten()
    xy = xy.flatten()
    return xx, xy

def spatial_feature_extract(feat_result, feat_content, xx, xy):
    l2, l3 = [], []
    device = feat_result[0].device

    # Loop over each extracted layer
    for i in range(len(feat_result)):
        fr = feat_result[i]
        fc = feat_content[i]

        # Hack to detect reduced scale
        if i > 0 and feat_result[i-1].size(2) > feat_result[i].size(2):
            xx = xx/2.0
            xy = xy/2.0

        # Go back to ints and get residual
        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        # Do bilinear resampling
        w00 = torch.from_numpy((1.-xxr)*(1.-xyr)).float().view(1, 1, -1, 1).to(device)
        w01 = torch.from_numpy((1.-xxr)*xyr).float().view(1, 1, -1, 1).to(device)
        w10 = torch.from_numpy(xxr*(1.-xyr)).float().view(1, 1, -1, 1).to(device)
        w11 = torch.from_numpy(xxr*xyr).float().view(1, 1, -1, 1).to(device)

        xxm = np.clip(xxm.astype(np.int32), 0, fr.size(2)-1)
        xym = np.clip(xym.astype(np.int32), 0, fr.size(3)-1)

        s00 = xxm*fr.size(3) + xym
        s01 = xxm*fr.size(3) + np.clip(xym+1, 0, fr.size(3)-1)
        s10 = np.clip(xxm+1, 0, fr.size(2)-1)*fr.size(3) + xym
        s11 = np.clip(xxm+1, 0, fr.size(2)-1)*fr.size(3) + np.clip(xym+1, 0, fr.size(3)-1)

        fr = fr.view(1, fr.size(1), fr.size(2)*fr.size(3), 1)
        fr = fr[:,:,s00,:].mul_(w00).add_(fr[:,:,s01,:].mul_(w01)).add_(fr[:,:,s10,:].mul_(w10)).add_(fr[:,:,s11,:].mul_(w11))

        fc = fc.view(1, fc.size(1), fc.size(2)*fc.size(3), 1)
        fc = fc[:,:,s00,:].mul_(w00).add_(fc[:,:,s01,:].mul_(w01)).add_(fc[:,:,s10,:].mul_(w10)).add_(fc[:,:,s11,:].mul_(w11))

        l2.append(fr)
        l3.append(fc)

    x_st = torch.cat([li.contiguous() for li in l2], 1)
    c_st = torch.cat([li.contiguous() for li in l3], 1)

    return x_st, c_st
