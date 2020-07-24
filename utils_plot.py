
# https://github.com/sunniesuhyoung/DST

# Functions for plotting intermediate and/or extra images

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils_misc import pil_loader
from warp import apply_warp

def convert_image(x):
    x_out = np.clip(x.permute(1,2,0).detach().cpu().numpy(), -1.0, 1.0)
    x_out -= x_out.min()
    x_out /= x_out.max()
    x_out = (x_out*255).astype(np.uint8)
    return x_out

def plot_intermediate(new_im, content_im_warp, output_dir, output_prefix, colors, down_fac,
                        src_Kpts, thetas_Kpts, target_Kpts, scale, i):

    if not os.path.exists(output_dir + '/intermediate'):
        os.makedirs(output_dir + '/intermediate')

    # Renormalize images
    new_im_out = convert_image(new_im[0])
    content_im_warp_out = convert_image(content_im_warp[0])

    # Process points
    dst_Kpts = (src_Kpts + thetas_Kpts)/down_fac
    dst_Kpts[:,0] = torch.clamp(dst_Kpts[:,0], min=0., max=content_im_warp.size(2))
    dst_Kpts[:,1] = torch.clamp(dst_Kpts[:,1], min=0., max=content_im_warp.size(3))
    dst_x = np.divide(dst_Kpts[:,1].detach().cpu().numpy(), 1)
    dst_y = np.divide(dst_Kpts[:,0].detach().cpu().numpy(), 1)
    src_x = np.divide(src_Kpts[:,1].detach().cpu().numpy(), down_fac)
    src_y = np.divide(src_Kpts[:,0].detach().cpu().numpy(), down_fac)
    target_x = np.divide(target_Kpts[:,1].detach().cpu().numpy(), down_fac)
    target_y = np.divide(target_Kpts[:,0].detach().cpu().numpy(), down_fac)

    # Stylized output marked
    plt.clf()
    plt.imshow(new_im_out)
    for j in range(src_x.shape[0]):
        plt.plot(dst_x[j], dst_y[j], marker='*', color=colors[j], markersize=5)
        plt.plot(src_x[j], src_y[j], marker='o', color=colors[j], markersize=5)
        plt.plot(target_x[j], target_y[j], marker='s', color=colors[j], markersize=5)
        plt.arrow(src_x[j], src_y[j], dst_x[j]-src_x[j], dst_y[j]-src_y[j], color=colors[j], width=0.05)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/intermediate/' + output_prefix + '_DSToutput_scl{}_iter{}.png'.format(scale+1, i), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Warped content marked
    plt.clf()
    plt.imshow(content_im_warp_out)
    for j in range(src_x.shape[0]):
        plt.plot(dst_x[j], dst_y[j], marker='*', color=colors[j], markersize=5)
        plt.plot(src_x[j], src_y[j], marker='o', color=colors[j], markersize=5)
        plt.plot(target_x[j], target_y[j], marker='s', color=colors[j], markersize=5)
        plt.arrow(src_x[j], src_y[j], dst_x[j]-src_x[j], dst_y[j]-src_y[j], color=colors[j], width=0.05)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/intermediate/' + output_prefix + '_contentwarped_scl{}_iter{}.png'.format(scale+1, i), bbox_inches='tight', pad_inches=0)
    plt.close()



def save_plots(im_size, curr_im, new_im, content_im, style_im, output_dir, output_prefix, style_path, style_pts_path, colors,
                src_Kpts, src_Kpts_aug, dst_Kpts, dst_Kpts_aug, target_Kpts, target_Kpts_o, border_Kpts, device):

    if not os.path.exists(output_dir + '/plots'):
        os.makedirs(output_dir + '/plots')

    # Warp content image to the learned destination points
    content_im_warp = content_im.clone()
    content_im_warp, _ = apply_warp(content_im_warp, [src_Kpts_aug], [dst_Kpts_aug], device, sharp=True)

    # Naively warp content and DST texture-stylized images from source to target
    target_Kpts_aug = torch.cat([target_Kpts, border_Kpts], 0)
    sizes = torch.FloatTensor([curr_im.size(2), curr_im.size(3)]).to(device)
    target_Kpts_aug = target_Kpts_aug/sizes
    target_Kpts_aug = torch.clamp(target_Kpts_aug, min=0., max=1.)
    output_targetwarp = curr_im.clone()
    content_targetwarp = content_im.clone()
    output_targetwarp, content_targetwarp, _ = apply_warp(output_targetwarp, [src_Kpts_aug], [target_Kpts_aug], device, sharp=True, im2=content_targetwarp)

    # Convert all images to numpy arrays
    content_im_out = convert_image(content_im[0]) # Content image
    style_im_out = convert_image(style_im[0]) # Style image
    new_im_out = convert_image(new_im[0]) # DST output (stylized + warped)
    content_im_warp_out = convert_image(content_im_warp[0]) # Content warped
    output_targetwarp_out = convert_image(output_targetwarp[0]) # Stylized naively warped
    content_targetwarp_out = convert_image(content_targetwarp[0]) # Content naively warped

    # Convert keypoints into numpy arrays
    src_center = src_Kpts.mean(0)
    src_x = src_Kpts[:,1].detach().cpu().numpy()
    src_y = src_Kpts[:,0].detach().cpu().numpy()
    target_x = target_Kpts[:,1].detach().cpu().numpy()
    target_y = target_Kpts[:,0].detach().cpu().numpy()
    dx = (target_Kpts[:,1] - src_Kpts[:,1]).detach().cpu().numpy()
    dy = (target_Kpts[:,0] - src_Kpts[:,0]).detach().cpu().numpy()
    dst_x = dst_Kpts[:,1].detach().cpu().numpy()
    dst_y = dst_Kpts[:,0].detach().cpu().numpy()

    # Produce plots
    plot_naively_warped_stylized(output_dir, output_prefix, colors, output_targetwarp_out,
                                    src_center, src_x, src_y, target_x, target_y)
    plot_naively_warped_content(output_dir, output_prefix, colors, content_targetwarp_out,
                                src_center, src_x, src_y, target_x, target_y)
    plot_DSToutput(output_dir, output_prefix, colors, new_im_out,
                    src_center, src_x, src_y, target_x, target_y, dst_x, dst_y)
    plot_warped_content(output_dir, output_prefix, colors, content_im_warp_out,
                        src_center, src_x, src_y, target_x, target_y, dst_x, dst_y)
    plot_content(output_dir, output_prefix, colors, content_im_out, src_x, src_y)
    plot_style(output_dir, output_prefix, colors, style_path, style_pts_path)

def plot_naively_warped_stylized(output_dir, output_prefix, colors, output_targetwarp_out,
                                    src_center, src_x, src_y, target_x, target_y):

    # DST texture-stylized image naively warped (source -> target)
    plt.clf()
    plt.imshow(output_targetwarp_out)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_stylized_naivewarp.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # DST texture-stylized image naively warped (source -> target) with points and arrows
    plt.clf()
    plt.imshow(output_targetwarp_out)
    # plt.plot(src_center.detach().cpu().numpy()[1], src_center.detach().cpu().numpy()[0], marker='o', color='k', markersize=5)
    for j in range(src_x.shape[0]):
        plt.plot(src_x[j], src_y[j], marker='o', color=colors[j], markersize=5)
        plt.plot(target_x[j], target_y[j], marker='s', color=colors[j], markersize=5)
        plt.arrow(src_x[j], src_y[j], target_x[j]-src_x[j], target_y[j]-src_y[j], color=colors[j], width=0.05)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_stylized_naivewarp_marked.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_naively_warped_content(output_dir, output_prefix, colors, content_targetwarp_out,
                                src_center, src_x, src_y, target_x, target_y):

    # Naively warped content image (source -> target)
    plt.clf()
    plt.imshow(content_targetwarp_out)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_content_naivewarp.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Naively warped content image (source -> target) with points and arrows
    plt.clf()
    plt.imshow(content_targetwarp_out)
    # plt.plot(src_center.detach().cpu().numpy()[1], src_center.detach().cpu().numpy()[0], marker='o', color='k', markersize=5)
    for j in range(src_x.shape[0]):
        plt.plot(src_x[j], src_y[j], marker='o', color=colors[j], markersize=5)
        plt.plot(target_x[j], target_y[j], marker='s', color=colors[j], markersize=5)
        plt.arrow(src_x[j], src_y[j], target_x[j]-src_x[j], target_y[j]-src_y[j], color=colors[j], width=0.05)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_content_naivewarp_marked.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_DSToutput(output_dir, output_prefix, colors, new_im_out,
                    src_center, src_x, src_y, target_x, target_y, dst_x, dst_y):

    # Stylized output image
    plt.clf()
    plt.imshow(new_im_out)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_DSToutput.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Stylized output image with points and arrows
    plt.clf()
    plt.imshow(new_im_out)
    # plt.plot(src_center.detach().cpu().numpy()[1], src_center.detach().cpu().numpy()[0], marker='o', color='k', markersize=5)
    for j in range(src_x.shape[0]):
        plt.plot(dst_x[j], dst_y[j], marker='*', color=colors[j], markersize=8)
        plt.plot(src_x[j], src_y[j], marker='o', color=colors[j], markersize=5)
        plt.plot(target_x[j], target_y[j], marker='s', color=colors[j], markersize=5)
        plt.arrow(src_x[j], src_y[j], dst_x[j]-src_x[j], dst_y[j]-src_y[j], color=colors[j], width=0.05)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_DSToutput_marked.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_warped_content(output_dir, output_prefix, colors, content_im_warp_out,
                        src_center, src_x, src_y, target_x, target_y, dst_x, dst_y):

    # Warped content image
    plt.clf()
    plt.imshow(content_im_warp_out)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_content_warped.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Warped content image with points and arrows
    plt.clf()
    plt.imshow(content_im_warp_out)
    # plt.plot(src_center.detach().cpu().numpy()[1], src_center.detach().cpu().numpy()[0], marker='o', color='k', markersize=5)
    for j in range(src_x.shape[0]):
        plt.plot(dst_x[j], dst_y[j], marker='*', color=colors[j], markersize=8)
        plt.plot(src_x[j], src_y[j], marker='o', color=colors[j], markersize=5)
        plt.plot(target_x[j], target_y[j], marker='s', color=colors[j], markersize=5)
        plt.arrow(src_x[j], src_y[j], dst_x[j]-src_x[j], dst_y[j]-src_y[j], color=colors[j], width=0.05)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_content_warped_marked.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_content(output_dir, output_prefix, colors, content_im_out, src_x, src_y):

    # Original content image
    plt.clf()
    plt.imshow(content_im_out)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_content.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Content image with source points
    plt.clf()
    plt.imshow(content_im_out)
    for j in range(src_x.shape[0]):
        plt.plot(src_x[j], src_y[j], marker='o', color=colors[j], markersize=5)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_content_marked.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_style(output_dir, output_prefix, colors, style_path, style_pts_path):

    style_im_out = pil_loader(style_path)
    raw_target_pts = np.loadtxt(style_pts_path, delimiter=',')
    raw_target_center = np.mean(raw_target_pts, axis=0)

    # Original style image
    plt.clf()
    plt.imshow(style_im_out)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_style.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Style image with keypoints
    plt.clf()
    plt.imshow(style_im_out)
    for j in range(len(colors)):
        plt.plot(raw_target_pts[j,1], raw_target_pts[j,0], marker='s', color=colors[j], markersize=5)
    plt.axes().set_aspect('equal')
    plt.axes().set_axis_off()
    plt.tight_layout()
    plt.savefig(output_dir + '/plots/' + output_prefix + '_style_marked.png', bbox_inches='tight', pad_inches=0)
    plt.close()
