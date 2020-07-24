
# https://github.com/sunniesuhyoung/DST

import os
import sys
import time
import torch
import numpy as np
from PIL import Image

from styletransfer import DST
from vggfeatures import VGG16_Extractor
from utils_plot import convert_image
from utils_misc import pil_loader, pil_resize_long_edge_to, pil_to_tensor

# Parse Arguments
content_path = str(sys.argv[1])
style_path = str(sys.argv[2])
content_pts_path = str(sys.argv[3])
style_pts_path = str(sys.argv[4])
output_dir = str(sys.argv[5])
output_prefix = str(sys.argv[6])
im_size = int(sys.argv[7])
max_iter = int(sys.argv[8])
checkpoint_iter = int(sys.argv[9])
content_weight = float(sys.argv[10]) # alpha
warp_weight = float(sys.argv[11]) # beta
reg_weight = float(sys.argv[12]) # gamma
optim = str(sys.argv[13])
lr = float(sys.argv[14])
verbose = int(sys.argv[15]) > 0
save_intermediate = int(sys.argv[16]) > 0
save_extra = int(sys.argv[17]) > 0
device = str(sys.argv[18])

# Print settings
print('\n\n---------------------------------')
print('Started Deformable Style Transfer')
print('---------------------------------')

print('\nSettings')
print('   content_path:', content_path)
print('   style_path:', style_path)
print('   content_pts_path:', content_pts_path)
print('   style_pts_path:', style_pts_path)
print('   output_dir:', output_dir)
print('   output_prefix:', output_prefix)
print('   im_size:', im_size)
print('   max_iter:', max_iter)
print('   checkpoint_iter:', checkpoint_iter)
print('   content_weight:', content_weight)
print('   warp_weight:', warp_weight)
print('   reg_weight:', reg_weight)
print('   optim:', optim)
print('   lr:', lr)
print('   verbose:', verbose)
print('   save_intermediate:', save_intermediate)
print('   save_extra:', save_extra)

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define feature extractor
extractor = VGG16_Extractor().to(device)

# Load content/style images and keypoints
content_pil = pil_loader(content_path)
style_pil = pil_loader(style_path)
content_pts = np.loadtxt(content_pts_path, delimiter=',')
style_pts = np.loadtxt(style_pts_path, delimiter=',')

# Rescale images
content_resized = pil_resize_long_edge_to(content_pil, im_size)
style_resized = pil_resize_long_edge_to(style_pil, im_size)
content_im_orig = pil_to_tensor(content_resized).to(device)
style_im_orig = pil_to_tensor(style_resized).to(device)

# Rescale points (assuming that points are in the original image's scale)
c_width, c_height = content_pil.size
c_fac = im_size/max(c_width, c_height)
for i in range(content_pts.shape[0]):
    content_pts[i][0] *= c_fac
    content_pts[i][1] *= c_fac

s_width, s_height = style_pil.size
s_fac = im_size/max(s_width, s_height)
for i in range(style_pts.shape[0]):
    style_pts[i][0] *= s_fac
    style_pts[i][1] *= s_fac

content_pts = torch.from_numpy(content_pts).float()
style_pts = torch.from_numpy(style_pts).float()

# Initialize the output image as the content image (This is a simpler initialization
# than what's described in the STROTSS paper, but we found that results are similar)
initial_im = content_im_orig.clone()

# Run deformable style transfer
start_time = time.time()
output = DST(initial_im, content_im_orig, style_im_orig, extractor,
                content_path, style_path, content_pts, style_pts, style_pts_path,
                output_dir, output_prefix,
                im_size=im_size,
                max_iter=max_iter,
                checkpoint_iter=checkpoint_iter,
                content_weight=content_weight,
                warp_weight=warp_weight,
                reg_weight=reg_weight,
                optim=optim,
                lr=lr,
                verbose=verbose,
                save_intermediate=save_intermediate,
                save_extra=save_extra,
                device=device)

# Write the stylized output image
save_im = convert_image(output[0])
save_im = Image.fromarray(save_im)
save_im.save(output_dir + '/' + output_prefix + '.png')
# imwrite(output_dir + '/' + output_prefix + '.png', save_im)
print('\nSaved the stylized image at', output_dir + '/' + output_prefix + '.png')

# Report total time
end_time = time.time()
total_time = (end_time - start_time) / 60
print('\nFinished after {:04.3f} minutes\n'.format(total_time))
