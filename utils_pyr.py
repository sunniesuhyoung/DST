
# Functions for converting pixel images <-> laplacian pyramids

# Code from https://github.com/futscdav/strotss

import numpy as np
import torch
import torch.nn.functional as F

def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)

def laplacian(x): # x - upsample(downsample(x))
    return x - tensor_resample(tensor_resample(x, [max(x.size(2)//2, 1), max(x.size(3)//2, 1)]), [x.size(2), x.size(3)])

def dec_lap_pyr(x, levels):
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(current, [max(current.size(2)//2, 1), max(current.size(3)//2, 1)])
    pyramid.append(current)
    return pyramid

def syn_lap_pyr(pyr):
    current = pyr[-1]
    for i in range(len(pyr)-2, -1, -1): # iterate from len-2 to 0
        up_h, up_w = pyr[i].size(2), pyr[i].size(3)
        current = pyr[i] + tensor_resample(current, (up_h, up_w))
    return current
