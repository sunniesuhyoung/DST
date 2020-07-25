
# https://github.com/sunniesuhyoung/DST

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from loss import content_loss, remd_loss, moment_loss, TV, pairwise_distances_sq_l2
from warp import apply_warp
from utils_pyr import syn_lap_pyr, dec_lap_pyr
from utils_keypoints import init_keypoint_params, gen_dst_pts_keypoints
from utils_misc import sample_indices, spatial_feature_extract
from utils_save import save_loss, save_points
from utils_plot import save_plots, plot_intermediate

def DST(input_im, content_im, style_im, extractor, content_path, style_path,
        content_pts, style_pts, style_pts_path, output_dir, output_prefix,
        im_size = 256,
        max_iter = 250,
        checkpoint_iter = 50,
        content_weight = 8.,
        warp_weight = 0.3,
        reg_weight = 10,
        scales = 3,
        pyr_levs = 5,
        sharp_warp = False,
        optim = 'adam',
        lr = 1e-3,
        warp_lr_fac = 1.,
        verbose = False,
        save_intermediate = False,
        save_extra = False,
        device = 'cuda:0'):

    # If warp weight is 0, run the base method STROTSS
    use_DST = True
    if warp_weight == 0.:
        use_DST = False

    # Initialize warp parameters
    src_Kpts, target_Kpts, border_Kpts, no_flow_Kpts= init_keypoint_params(input_im, content_path, content_pts, style_pts, device)
    thetas_Kpts = Variable(torch.rand_like(src_Kpts).data*1e-4, requires_grad=True)

    # Clamp the target points so that they don't go outside the boundary
    target_Kpts[:,0] = torch.clamp(target_Kpts[:,0], min=5, max=content_im.size(2)-5)
    target_Kpts[:,1] = torch.clamp(target_Kpts[:,1], min=5, max=content_im.size(3)-5)
    target_Kpts_o = target_Kpts.clone().detach()

    # Assign colors to each set of points (used for visualization only)
    np.random.seed(1)
    colors = []
    for j in range(src_Kpts.shape[0]):
        colors.append(np.random.random(size=3))

    # Initialize pixel parameters
    s_pyr = dec_lap_pyr(input_im, pyr_levs)
    s_pyr = [Variable(li.data, requires_grad=True) for li in s_pyr]

    # Define parameters to be optimized
    s_pyr_list = [{'params': si} for si in s_pyr]
    if use_DST:
        thetas_opt_list = [{'params': thetas_Kpts, 'lr': lr*warp_lr_fac}]
    else:
        thetas_opt_list = []

    # Construct optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(s_pyr_list + thetas_opt_list, lr=lr, momentum=0.9)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(s_pyr_list + thetas_opt_list, lr=lr)
    else:
        optimizer = torch.optim.Adam(s_pyr_list + thetas_opt_list, lr=lr)

    # Set scales
    scale_list = list(range(scales))
    if scales == 1:
        scale_list = [0]

    # Create lists to store various loss values
    ell_list = []
    ell_style_list = []
    ell_content_list = []
    ell_warp_list = []
    ell_warp_TV_list = []

    # Iteratively stylize over more levels of image pyramid
    for scale in scale_list:

        down_fac = 2**(scales-1-scale)
        begin_ind = (scales-1-scale)
        content_weight_scaled = content_weight*down_fac

        print('\nOptimizing at scale {}, image size ({}, {})'.format(scale+1, content_im.size(2)//down_fac, content_im.size(3)//down_fac))

        if down_fac > 1.:
            content_im_scaled = F.interpolate(content_im, (content_im.size(2)//down_fac, content_im.size(3)//down_fac), mode='bilinear')
            style_im_scaled = F.interpolate(style_im, (style_im.size(2)//down_fac, style_im.size(3)//down_fac), mode='bilinear')
        else:
            content_im_scaled = content_im.clone()
            style_im_scaled = style_im.clone()

        # Compute feature maps that won't change for this scale
        with torch.no_grad():
            feat_content = extractor(content_im_scaled)

            feat_style = None
            for i in range(5):
                with torch.no_grad():
                    feat_e = extractor.forward_samples_hypercolumn(style_im_scaled, samps=1000)
                    feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

            feat_max = 3 + 2*64 + 2*128 + 3*256 + 2*512 # 2179 = sum of all extracted channels
            spatial_style = feat_style.view(1, feat_max, -1, 1)

            xx, xy = sample_indices(feat_content[0], feat_style)


        # Begin optimization for this scale
        for i in range(max_iter):

            optimizer.zero_grad()

            # Get current stylized image from the laplacian pyramid
            curr_im = syn_lap_pyr(s_pyr[begin_ind:])
            new_im = curr_im.clone()
            content_im_warp = content_im_scaled.clone()

            # Generate destination points with the current thetas
            src_Kpts_aug, dst_Kpts_aug, flow_Kpts_aug = gen_dst_pts_keypoints(src_Kpts, thetas_Kpts, no_flow_Kpts, border_Kpts)

            # Calculate warp loss
            ell_warp = torch.norm(target_Kpts_o - dst_Kpts_aug[:target_Kpts.size(0)], dim=1).mean()

            # Scale points to [0-1]
            src_Kpts_aug = src_Kpts_aug/torch.max(src_Kpts_aug, 0, keepdim=True)[0]
            dst_Kpts_aug = dst_Kpts_aug/torch.max(dst_Kpts_aug, 0, keepdim=True)[0]
            dst_Kpts_aug = torch.clamp(dst_Kpts_aug, min=0., max=1.)

            # Warp
            new_im, content_im_warp, warp_field = apply_warp(new_im, [src_Kpts_aug], [dst_Kpts_aug], device, sharp=sharp_warp, im2=content_im_warp)
            new_im = new_im.to(device)

            # Calculate total variation
            ell_warp_TV = TV(warp_field)

            # Extract VGG features of warped and unwarped stylized images
            feat_result_warped = extractor(new_im)
            feat_result_unwarped = extractor(curr_im)

            # Sample features to calculate losses with
            n = 2048
            if i % 1 == 0 and i != 0:
                np.random.shuffle(xx)
                np.random.shuffle(xy)
            spatial_result_warped, spatial_content = spatial_feature_extract(feat_result_warped, feat_content, xx[:n], xy[:n])
            spatial_result_unwarped, _ = spatial_feature_extract(feat_result_unwarped, feat_content, xx[:n], xy[:n])

            # Content loss
            ell_content = content_loss(spatial_result_unwarped, spatial_content)

            # Style loss

            # Lstyle(Unwarped X, S)
            loss_remd1 = remd_loss(spatial_result_unwarped, spatial_style, cos_d=True)
            loss_moment1 = moment_loss(spatial_result_unwarped, spatial_style, moments=[1,2])
            loss_color1 = remd_loss(spatial_result_unwarped[:,:3,:,:], spatial_style[:,:3,:,:], cos_d=False)
            loss_style1 = loss_remd1 + loss_moment1 + (1./max(content_weight_scaled, 1.))*loss_color1

            # Lstyle(Warped X, S)
            loss_remd2 = remd_loss(spatial_result_warped, spatial_style, cos_d=True)
            loss_moment2 = moment_loss(spatial_result_warped, spatial_style, moments=[1,2])
            loss_color2 = remd_loss(spatial_result_warped[:,:3,:,:], spatial_style[:,:3,:,:], cos_d=False)
            loss_style2 = loss_remd2 + loss_moment2 + (1./max(content_weight_scaled, 1.))*loss_color2

            # Total loss
            if use_DST:
                ell_style = loss_style1 + loss_style2
                ell = content_weight_scaled*ell_content + ell_style + warp_weight*ell_warp + reg_weight*ell_warp_TV
            else:
                ell_style = loss_style1
                ell = content_weight_scaled*ell_content + ell_style

            # Record loss values
            ell_list.append(ell.item())
            ell_content_list.append(ell_content.item())
            ell_style_list.append(ell_style.item())
            ell_warp_list.append(ell_warp.item())
            ell_warp_TV_list.append(ell_warp_TV.item())

            # Output intermediate loss
            if i==0 or i%checkpoint_iter == 0:
                print('   STEP {:03d}: Loss {:04.3f}'.format(i, ell))
                if verbose:
                    print('             = alpha*Lcontent {:04.3f}'.format(content_weight_scaled*ell_content))
                    print('               + Lstyle {:04.3f}'.format(ell_style))
                    print('               + beta*Lwarp {:04.3f}'.format(warp_weight*ell_warp))
                    print('               + gamma*TV {:04.3f}'.format(reg_weight*ell_warp_TV))
                if save_intermediate:
                    plot_intermediate(new_im, content_im_warp, output_dir, output_prefix, colors,
                                        down_fac, src_Kpts, thetas_Kpts, target_Kpts, scale, i)

            # Take a gradient step
            ell.backward()
            optimizer.step()


    # Optimization finished
    src_Kpts_aug, dst_Kpts_aug, flow_Kpts_aug = gen_dst_pts_keypoints(src_Kpts, thetas_Kpts, no_flow_Kpts, border_Kpts)
    sizes = torch.FloatTensor([new_im.size(2), new_im.size(3)]).to(device)
    src_Kpts_aug = src_Kpts_aug/sizes
    dst_Kpts_aug = dst_Kpts_aug/sizes
    dst_Kpts_aug = torch.clamp(dst_Kpts_aug, min=0., max=1.)
    dst_Kpts = dst_Kpts_aug[:src_Kpts.size(0)]

    # Apply final warp
    sharp_final = True
    new_im = curr_im.clone()
    content_im_warp = content_im.clone()
    new_im, _ = apply_warp(new_im, [src_Kpts_aug], [dst_Kpts_aug], device, sharp=sharp_final)

    # Optionally save loss, keypoints, and optimized warp parameter thetas
    if save_extra:
        save_plots(im_size, curr_im, new_im, content_im, style_im, output_dir, output_prefix, style_path, style_pts_path, colors,
                    src_Kpts, src_Kpts_aug, dst_Kpts*sizes, dst_Kpts_aug, target_Kpts, target_Kpts_o, border_Kpts, device)
        save_loss(output_dir, output_prefix, content_weight, warp_weight, reg_weight, max_iter, scale_list,
                    ell_list, ell_style_list, ell_content_list, ell_warp_list, ell_warp_TV_list)
        save_points(output_dir, output_prefix, src_Kpts, dst_Kpts*sizes, src_Kpts_aug*sizes,
                    dst_Kpts_aug*sizes, target_Kpts, thetas_Kpts)

    # Return the stylized output image
    return new_im
