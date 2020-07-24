
# https://github.com/sunniesuhyoung/DST

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from PIL import Image

from warp import umeyama

print('\n\n--------------------------')
print('Started Cleaning Keypoints')
print('--------------------------')
start_time = time.time()

# Parse arguments
content_path = str(sys.argv[1])
style_path = str(sys.argv[2])
content_pts_path = str(sys.argv[3])
style_pts_path = str(sys.argv[4])
activation_path = str(sys.argv[5])
output_path = str(sys.argv[6])
im_size = int(sys.argv[7])
NBB = int(sys.argv[8]) > 0
max_num_points = int(sys.argv[9])
b = int(sys.argv[10])

print('\nSettings')
print('   content_path:', content_path)
print('   style_path:', style_path)
print('   content_pts_path:', content_pts_path)
print('   style_pts_path:', style_pts_path)
print('   activation_path:', activation_path)
print('   output_path:', output_path)
print('   NBB:', NBB)
print('   im_size:', im_size)
print('   max_num_points:', max_num_points)
print('   b:', b)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Assign colors to each set of points (used for visualization only)
np.random.seed(17)
colors = []
for j in range(max_num_points):
    colors.append(np.random.random(size=3))

# Load points
A_corres = np.loadtxt(content_pts_path, delimiter=',')
B_corres = np.loadtxt(style_pts_path, delimiter=',')
try:
    activations = np.loadtxt(activation_path, delimiter=',')
except:
    print('Activation values not provided')
    activations = np.ones(A_corres.shape[0])

print('\nStarted with {} points'.format(A_corres.shape[0]))

# Load images
A_im_orig = Image.open(content_path)
A_width, A_height = A_im_orig.size
A_fac = im_size/max(A_width, A_height)
A_newwidth = int(A_width * A_fac)
A_newheight = int(A_height * A_fac)
A_im = np.array(A_im_orig.resize((A_newwidth, A_newheight)))

B_im_orig = Image.open(style_path)
B_width, B_height = B_im_orig.size
B_fac = im_size/max(B_width, B_height)
B_newwidth = int(B_width * B_fac)
B_newheight = int(B_height * B_fac)
B_im = np.array(B_im_orig.resize((B_newwidth, B_newheight)))

# If input points are from NBB, do NBB-specific pre-processing
if NBB:
    if True:
        plt.clf()
        plt.imshow(A_im)
        for i in range(A_corres.shape[0]):
            plt.plot(A_corres[i][1], A_corres[i][0], marker='o', markersize=10)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(output_path + '/A_all_correspondence.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    if True:
        plt.clf()
        plt.imshow(B_im)
        for i in range(B_corres.shape[0]):
            plt.plot(B_corres[i][1], B_corres[i][0], marker='o', markersize=10)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(output_path + '/B_all_correspondence.png', bbox_inches='tight', pad_inches=-0.1)
        plt.close()

    # Remove points with activations less than 1
    lowactv = np.where(activations < 1)
    A_corres = np.delete(A_corres, lowactv, axis=0)
    B_corres = np.delete(B_corres, lowactv, axis=0)
    activations = np.delete(activations, lowactv, axis=0)
    print('\n{} points removed because their activations were less than 1 (NBB-specific processing)'.format(lowactv[0].shape[0]))

    # Select n points in a greedy way
    A_selected = np.zeros((max_num_points, 2), dtype=np.int32)
    B_selected = np.zeros((max_num_points, 2), dtype=np.int32)
    act_selected = np.zeros(max_num_points, dtype=np.float32)

    for i in range(max_num_points):

        # Add the point that has the highest activation value
        idx = np.argmax(activations, axis=0)
        A_r, A_c = A_corres[idx]
        B_r, B_c = B_corres[idx]

        A_selected[i, 0] = int(A_r)
        A_selected[i, 1] = int(A_c)
        B_selected[i, 0] = int(B_r)
        B_selected[i, 1] = int(B_c)
        act_selected[i] = activations[idx]

        # Zero-out activation values of points that are within 'b' space of the selected point
        rleft = max(0, A_r-b)
        rright = min(A_height, A_r+b)
        cleft = max(0, A_c-b)
        cright = min(A_width, A_c+b)

        rcondition = np.logical_and(A_corres[:, 0] >= rleft, A_corres[:, 0] <= rright)
        ccondition = np.logical_and(A_corres[:, 1] >= cleft, A_corres[:, 1] <= cright)
        condition = np.logical_and(rcondition, ccondition)
        activations[condition] *= 0.

    # Remove points that are redundant
    A_prev = [0., 0.]; A_curr = [0., 0.]; B_prev = [0., 0.]; B_curr = [0., 0.]
    repeats = []
    for i in range(max_num_points):
        A_curr = A_selected[i]
        B_curr = B_selected[i]
        if np.logical_and(np.all(A_curr == A_prev), np.all(B_curr == B_prev)):
            repeats.append(i)
        A_prev = A_selected[i]
        B_prev = B_selected[i]

    A_selected = np.delete(A_selected, repeats, axis=0)
    B_selected = np.delete(B_selected, repeats, axis=0)
    act_selected = np.delete(act_selected, repeats, axis=0)

    print('\n{} points selected in a greedy way (NBB-specific processing)'.format(A_selected.shape[0]))

# Derive target points from the style image's keypoints via a linear similarity transformation
def get_target_pts(A_selected, B_selected):
    src_pts = torch.from_numpy(A_selected).float()
    target_pts = torch.from_numpy(B_selected).float()

    T = umeyama(src=src_pts.data.cpu().numpy(), dst=target_pts.data.cpu().numpy(), estimate_scale=True)
    T = torch.from_numpy(T).float()

    target_pts_padded = torch.cat((target_pts, torch.ones((target_pts.size(0), 1))), 1)
    new_target_pts = torch.matmul(torch.inverse(T), torch.transpose(target_pts_padded, 0, 1))
    new_target_pts = torch.transpose(new_target_pts[:2], 0, 1)
    new_target_pts[:,0] = torch.clamp(new_target_pts[:,0], min=1, max=A_im.shape[0]-1)
    new_target_pts[:,1] = torch.clamp(new_target_pts[:,1], min=1, max=A_im.shape[1]-1)

    return new_target_pts.data.cpu().numpy()

# Optionally, remove crossing points to encourage a smoother warp field
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False

def remove_crossing(A_selected, B_selected, B_selected_new, act_selected):
    for i in range(A_selected.shape[0]):
        inds = []
        for j in range(i+1, A_selected.shape[0]):
            L1 = line(A_selected[i], B_selected_new[i])
            L2 = line(A_selected[j], B_selected_new[j])

            R = intersection(L1, L2)
            if R:
                i_xmin = min(A_selected[i][0], B_selected_new[i][0])
                i_xmax = max(A_selected[i][0], B_selected_new[i][0])
                j_xmin = min(A_selected[j][0], B_selected_new[j][0])
                j_xmax = max(A_selected[j][0], B_selected_new[j][0])

                i_ymin = min(A_selected[i][1], B_selected_new[i][1])
                i_ymax = max(A_selected[i][1], B_selected_new[i][1])
                j_ymin = min(A_selected[j][1], B_selected_new[j][1])
                j_ymax = max(A_selected[j][1], B_selected_new[j][1])

                if R[0] >= max(i_xmin, j_xmin) and R[0] <= min(i_xmax, j_xmax) and R[1] >= max(i_ymin, j_ymin) and R[1] <= min(i_ymax, j_ymax):
                    inds.append(j)

        A_selected = np.delete(A_selected, inds, axis=0)
        B_selected = np.delete(B_selected, inds, axis=0)
        B_selected_new = np.delete(B_selected_new, inds, axis=0)
        act_selected = np.delete(act_selected, inds, axis=0)

    return A_selected, B_selected, B_selected_new, act_selected

# Transform style points to get aligned target points, then remove crossing points
B_selected_new = get_target_pts(A_selected, B_selected)
A_selected, B_selected, B_selected_new, act_selected = remove_crossing(A_selected, B_selected, B_selected_new, act_selected)

# Repeat because there can be new crossing points after re-alignment
B_selected_new = get_target_pts(A_selected, B_selected)
A_selected, B_selected, B_selected_new, act_selected = remove_crossing(A_selected, B_selected, B_selected_new, act_selected)
print('\n{} points remaining after removing crossing points'.format(A_selected.shape[0]))

# Get final target points
B_selected_new = get_target_pts(A_selected, B_selected)

if True:
    plt.clf()
    plt.imshow(A_im)
    for i in range(A_selected.shape[0]):
        plt.plot(A_selected[i][1], A_selected[i][0], marker='o', markersize=10, color=colors[i])
        plt.plot(B_selected_new[i][1], B_selected_new[i][0], marker='s', markersize=10, color=colors[i])
        plt.arrow(A_selected[i][1], A_selected[i][0], B_selected_new[i][1]-A_selected[i][1], B_selected_new[i][0]-A_selected[i][0], color=colors[i], width=0.5)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path + '/A_source_and_target_final.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if True:
    plt.clf()
    plt.imshow(A_im)
    for i in range(A_selected.shape[0]):
        plt.plot(A_selected[i][1], A_selected[i][0], marker='o', markersize=10, color=colors[i])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path + '/A_selected_final.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if True:
    plt.clf()
    plt.imshow(B_im)
    for i in range(B_selected.shape[0]):
        plt.plot(B_selected[i][1], B_selected[i][0], marker='s', markersize=10, color=colors[i])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path + '/B_selected_final.png', bbox_inches='tight', pad_inches=0)
    plt.close()

# Scale points to match the original image's scale
for i in range(A_selected.shape[0]):
    A_selected[i][0] /= A_fac
    A_selected[i][1] /= A_fac

for i in range(B_selected.shape[0]):
    B_selected[i][0] /= B_fac
    B_selected[i][1] /= B_fac

# Output cleaned points
with open(output_path + '/correspondence_A.txt', 'wt') as opt_file:
    for j in range(A_selected.shape[0]):
        opt_file.write('%i, %i\n' % (A_selected[j][0], A_selected[j][1]))

with open(output_path + '/correspondence_B.txt', 'wt') as opt_file:
    for j in range(B_selected.shape[0]):
        opt_file.write('%i, %i\n' % (B_selected[j][0], B_selected[j][1]))

with open(output_path + '/correspondence_activation.txt', 'wt') as opt_file:
    for j in range(act_selected.shape[0]):
        opt_file.write('%f\n' % (act_selected[j]))

print('\nSaved cleaned content points at', output_path + '/correspondence_A.txt')
print('Saved cleaned style points at', output_path + '/correspondence_B.txt')
print('Saved cleaned activations at', output_path + '/correspondence_activation.txt')

# Finish and print time
end_time = time.time()
total_time = end_time - start_time
print('\nFinished after {:04.3f} seconds'.format(total_time))
