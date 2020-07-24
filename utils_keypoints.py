
# https://github.com/sunniesuhyoung/DST

import torch
import numpy as np

from warp import umeyama

def init_keypoint_params(input_im, content_path, content_pts, style_pts, device, border_num_pts=80):

    # Align points with linear similarity transformation
    T = umeyama(src=content_pts.data.cpu().numpy(), dst=style_pts.data.cpu().numpy(), estimate_scale=True)
    T = torch.from_numpy(T).float()
    target_pts_padded = torch.cat((style_pts, torch.ones((style_pts.size(0), 1))), 1)
    target_pts = torch.matmul(torch.inverse(T), torch.transpose(target_pts_padded, 0, 1))
    target_pts = torch.transpose(target_pts[:2], 0, 1)

    # Add fixed points at image borders to prevent weird warping
    height = input_im.size(2)
    width = input_im.size(3)

    w_d = width//(border_num_pts+1)
    w_pts = w_d*(np.arange(border_num_pts)+1)
    h_d = height//(border_num_pts+1)
    h_pts = h_d*(np.arange(border_num_pts)+1)

    border_pts = [[0, 0], [height-1, 0], [0, width-1], [height-1, width-1]]
    for i in range(border_num_pts):
        border_pts.append([h_pts[i], 0])
        border_pts.append([h_pts[i], width-1])
        border_pts.append([0, w_pts[i]])
        border_pts.append([height-1, w_pts[i]])
    border_pts = torch.from_numpy(np.asarray(border_pts)).float()

    no_flow = [[0., 0.]] * len(border_pts)
    no_flow = torch.from_numpy(np.asarray(no_flow)).float()

    return content_pts.to(device), target_pts.to(device), border_pts.to(device), no_flow.to(device)


def gen_dst_pts_keypoints(src_pts, thetas, no_flow, border_pts):

    flow_pts = thetas
    dst_pts = src_pts + flow_pts

    flow_pts_aug = torch.cat([flow_pts, no_flow], 0)
    src_pts_aug = torch.cat([src_pts, border_pts], 0)
    dst_pts_aug = torch.cat([dst_pts, border_pts], 0)

    return src_pts_aug, dst_pts_aug, flow_pts_aug
