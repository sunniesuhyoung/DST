
# Code modified from https://github.com/kfiraberman/neural_best_buddies

import os
import time
import numpy as np
from models import vgg19_model
from algorithms import neural_best_buddies as NBBs
from util import util
from util import MLS

from options.options import Options
opt = Options().parse()

vgg19 = vgg19_model.define_Vgg19(opt)
# save_dir = os.path.join(opt.results_dir, opt.name)
save_dir = os.path.join(opt.results_dir)

print('\n\n---------------------------')
print('Started Neural Best-Buddies')
print('---------------------------')
start_time = time.time()

nbbs = NBBs.sparse_semantic_correspondence(vgg19, opt.gpu_ids, opt.tau, opt.border_size, save_dir, opt.k_per_level, opt.k_final, opt.fast)
A = util.read_image(opt.datarootA, opt.imageSize)
B = util.read_image(opt.datarootB, opt.imageSize)
points = nbbs.run(A, B, opt.datarootA, opt.datarootB)

end_time = time.time()
total_time = end_time - start_time
print('\nFinished after {:04.3f} seconds'.format(total_time))
