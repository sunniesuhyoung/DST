
# Helper functions for calculating loss

# Code based on https://github.com/futscdav/strotss

import time
import numpy as np
import torch
import torch.nn.functional as F

def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    dist = 1.-torch.mm(x, y_t)/x_norm/y_norm
    return dist

def pairwise_distances_sq_l2(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)

def distmat(x, y, cos_d=True):
    if cos_d:
        M = pairwise_distances_cos(x, y)
    else:
        M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M

def rgb_to_yuv(rgb):
    C = torch.Tensor([[0.577350,0.577350,0.577350],
                        [-0.577350,0.788675,-0.211325],
                        [-0.577350,-0.211325,0.788675]]).to(rgb.device)
    yuv = torch.mm(C, rgb)
    return yuv

def content_loss(feat_result, feat_content):
    d = feat_result.size(1)
    X = feat_result.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Y = feat_content.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    Mx = distmat(X, X)
    Mx = Mx/Mx.sum(0, keepdim=True)
    My = distmat(Y, Y)
    My = My/My.sum(0, keepdim=True)
    d = torch.abs(Mx-My).mean() * X.shape[0]

    return d


def remd_loss(X, Y, cos_d=True):
    d = X.shape[1]

    if d == 3:
        X = rgb_to_yuv(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y = rgb_to_yuv(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
    else:
        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    CX_M = distmat(X, Y, cos_d=cos_d)

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)

    remd = torch.max(m1.mean(), m2.mean())

    return remd


def moment_loss(X, Y, moments=[1,2]):
    loss = 0.
    X = X.squeeze().t()
    Y = Y.squeeze().t()

    mu_x = torch.mean(X, 0, keepdim=True)
    mu_y = torch.mean(Y, 0, keepdim=True)
    mu_d = torch.abs(mu_x - mu_y).mean()

    if 1 in moments:
        loss = loss + mu_d

    if 2 in moments:
        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0]-1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0]-1)
        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = loss + D_cov

    return loss

def TV(x):
    ell =  torch.pow(torch.abs(x[:,:,1:,: ] - x[:,:,0:-1,:  ]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,: ,1:] - x[:,:,:  ,0:-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,1:,1:] - x[:,:, :-1, :-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:,:,1:,:-1] - x[:,:,:-1,1:]), 2).mean()
    ell /= 4.
    return ell
