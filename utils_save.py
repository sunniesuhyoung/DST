
# https://github.com/sunniesuhyoung/DST

# Functions for saving loss values and points

import os
import numpy as	np
import matplotlib.pyplot as plt

def save_loss(output_dir, output_prefix, content_weight, warp_weight, reg_weight, max_iter,
                scale_list, ell_list, ell_style_list, ell_content_list, ell_warp_list, ell_warp_TV_list):

    if not os.path.exists(output_dir + '/loss'):
        os.makedirs(output_dir + '/loss')

    with open(output_dir + '/loss/' + output_prefix + '_ell.txt', 'wt') as opt_file:
        for i in range(len(ell_list)):
            opt_file.write('%.6f\n' % (ell_list[i]))

    with open(output_dir + '/loss/' + output_prefix + '_ell_style.txt', 'wt') as opt_file:
        for i in range(len(ell_style_list)):
            opt_file.write('%.6f\n' % (ell_style_list[i]))

    with open(output_dir + '/loss/' + output_prefix + '_ell_content.txt', 'wt') as opt_file:
        for i in range(len(ell_content_list)):
            opt_file.write('%.6f\n' % (ell_content_list[i]))

    with open(output_dir + '/loss/' + output_prefix + '_ell_warp.txt', 'wt') as opt_file:
        for i in range(len(ell_warp_list)):
            opt_file.write('%.6f\n' % (ell_warp_list[i]))

    with open(output_dir + '/loss/' + output_prefix + '_ell_warp_TV.txt', 'wt') as opt_file:
        for i in range(len(ell_warp_TV_list)):
            opt_file.write('%.6f\n' % (ell_warp_TV_list[i]))

    alphas = []
    for scale in scale_list:
        down_fac = 2**(max(scale_list)-scale)
        content_weight_scaled = content_weight*down_fac
        alphas.append([content_weight_scaled] * max_iter)
    alphas = np.concatenate(alphas).ravel().tolist()

    plt.clf()
    plt.plot(ell_list, label='Total loss')
    plt.plot(np.multiply(ell_content_list, alphas), label='alpha * Lcontent')
    plt.plot(ell_style_list, label='Lstyle')
    plt.plot(np.multiply(ell_warp_list, warp_weight), label = 'beta * Lwarp')
    plt.plot(np.multiply(ell_warp_TV_list, reg_weight), label = 'gamma * TV')
    for scale in scale_list:
        if scale > 0:
            plt.axvline(x=max_iter*scale, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.savefig(output_dir + '/loss/' + output_prefix + '_loss.png')
    plt.close()


def save_points(output_dir, output_prefix, src_Kpts, dst_Kpts, src_Kpts_aug, dst_Kpts_aug, target_Kpts, thetas_Kpts):

    if not os.path.exists(output_dir + '/points'):
        os.makedirs(output_dir + '/points')

    with open(output_dir + '/points/' + output_prefix + '_src_Kpts.txt', 'wt') as opt_file:
        for i in range(src_Kpts.size(0)):
            opt_file.write('%.6f,%.6f\n' % (src_Kpts[i,0].data.cpu().numpy(), src_Kpts[i,1].data.cpu().numpy()))

    with open(output_dir + '/points/' + output_prefix + '_dst_Kpts.txt', 'wt') as opt_file:
        for i in range(dst_Kpts.size(0)):
            opt_file.write('%.6f,%.6f\n' % (dst_Kpts[i,0].data.cpu().numpy(), dst_Kpts[i,1].data.cpu().numpy()))

    with open(output_dir + '/points/' + output_prefix + '_src_Kpts_aug.txt', 'wt') as opt_file:
        for i in range(src_Kpts_aug.size(0)):
            opt_file.write('%.6f,%.6f\n' % (src_Kpts_aug[i,0].data.cpu().numpy(), src_Kpts_aug[i,1].data.cpu().numpy()))

    with open(output_dir + '/points/' + output_prefix + '_dst_Kpts_aug.txt', 'wt') as opt_file:
        for i in range(dst_Kpts_aug.size(0)):
            opt_file.write('%.6f,%.6f\n' % (dst_Kpts_aug[i,0].data.cpu().numpy(), dst_Kpts_aug[i,1].data.cpu().numpy()))

    with open(output_dir + '/points/' + output_prefix + '_target_Kpts.txt', 'wt') as opt_file:
        for i in range(target_Kpts.size(0)):
            opt_file.write('%.6f,%.6f\n' % (target_Kpts[i,0].data.cpu().numpy(), target_Kpts[i,1].data.cpu().numpy()))

    with open(output_dir + '/points/' + output_prefix + '_thetas_Kpts.txt', 'wt') as opt_file:
        for i in range(thetas_Kpts.size(0)):
            opt_file.write('%.6f,%.6f\n' % (thetas_Kpts[i,0].data.cpu().numpy(), thetas_Kpts[i,1].data.cpu().numpy()))
