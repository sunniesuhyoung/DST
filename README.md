# Deformable Style Transfer (DST)

### [Project Page](https://sunniesuhyoung.github.io/DST-page/) | [Paper](https://arxiv.org/abs/2003.11038) | [Demo](https://bit.ly/DST-demo) | [1 Minute Video](https://youtu.be/7qUzfcn6TPk) | [10 Minute Talk](https://youtu.be/mVU5tSxS4is) 

This repo provides the source code and demo of our paper: **Deformable Style Transfer (ECCV 2020)**.

It also contains a pytorch implementation of a differentiable warping module that uses thin-plate spline interpolation. This is a reimplementation of WarpGAN's tensorflow code.

```
@InProceedings{Kim20DST,
  author = {Sunnie S. Y. Kim and Nicholas Kolkin and Jason Salavon and Gregory Shakhnarovich},
  title = {Deformable Style Transfer},
  year = {2020},  
  booktitle = {European Conference on Computer Vision (ECCV)}  
}
```

## Dependencies

- Python 3 (e.g. conda create -n DST python=3.7.3)
- pytorch, torchvision, cudatoolkit, numpy, PIL, matplotlib, sklearn

## Demo

See ```demo_DST.ipynb``` for a demo of our work. You can also run this demo on [Google Colab](https://bit.ly/DST-demo).  
See ```demo_warp.ipynb``` for a short demo of the warping module we use to deform images.

## Usage

See ```job_example.sh``` for a single script that will produce a DST output, given a pair of content and style images, with default settings that we like.

### 1. Run NBB to get correspondences
Run NBB to find correspondences between images. This will take about a minute or two to run. If you already have matching points, you can skip this step.  
```
python NBB/main.py --results_dir ${results_dir} --imageSize ${im_size} --fast \
  --datarootA ${content_path} --datarootB ${style_path}
```

### 2. Clean (NBB) points
If you're cleaning NBB points, set **NBB** to 1. This script will filter out points with low activations and select up to **max_num_points** with at least **b** space between them. Then it will remove crossing points.

If you're cleaning non-NBB points, set **NBB** to 0, and this script will only remove crossing points. 

This will take a few seconds to run.
```
python cleanpoints.py ${content_path} ${style_path} ${content_pts_path} ${style_pts_path} \
  ${activation_path} ${output_path} ${im_size} ${NBB} ${max_num_points} ${b}
```

### 3. Run DST

Finally, run deformable style transfer. Stylization will take a few minutes on a GPU.

By default, DST will stylize the image at three scales (small to big) with **max_iter** iterations at each scale. Change **content_weight** (alpha) and **warp_weight** (beta) to control the relative importance of content preservation and deformation to stylization, and **reg_weight** (gamma) to control the amount of regularization on the deformation. We like using 8, 0.5, and 50, respectively.

Set **verbose** to 1 to get the individual loss term values during training.  
Set **save_intermediate** to 1 to get intermediate stylized images every **checkpoint_iter** iterations.  
Set **save_extra** to 1 to get additional plots (e.g. content/stylized images naively warped, content image with DST warp) with points and arrows marked.
```
python main.py ${content_path} ${style_path} ${content_pts_path} ${style_pts_path} \
  ${output_dir} ${output_prefix} ${im_size} ${max_iter} \
  ${checkpoint_iter} ${content_weight} ${warp_weight} ${reg_weight} ${optim} \
  ${lr} ${verbose} ${save_intermediate} ${save_extra} ${device}
```

## Acknowledgment
Our code is based on code from the following papers:
- Style Transfer by Relaxed Optimal Transport and Self-Similarity. Nicholas Kolkin, Jason Salavon and Gregory Shakhnarovich. CVPR 2019. [[paper]](https://arxiv.org/abs/1904.12785) [[code]](https://github.com/nkolkin13/STROTSS) [[David Futschik's implementation]](https://github.com/futscdav/strotss)
- WarpGAN: Automatic Caricature Generation. Yichun Shi, Debayan Deb and Anil K. Jain. CVPR 2019. [[paper]](https://arxiv.org/abs/1811.10100) [[code]](https://github.com/seasonSH/WarpGAN)
- Neural Best-Buddies: Sparse Cross-Domain Correspondence. Kfir Aberman, Jing Liao, Mingyi Shi, Dani Lischinski, Baoquan Chen and Daniel Cohen-Or. SIGGRAPH 2018. [[paper]](https://arxiv.org/abs/1805.04140) [[code]](https://github.com/kfiraberman/neural_best_buddies)

