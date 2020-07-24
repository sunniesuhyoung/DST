#!/bin/bash

source activate style

python3 main.py --name nonFace_06 --imageSize 256 --k_final 15 --fast \
  --datarootA /share/data/vision-greg2/users/sunnie/style/WarpImages2020/nonFace/content/06.jpg \
  --datarootB /share/data/vision-greg2/users/sunnie/style/WarpImages2020/nonFace/style/06.jpg  \
