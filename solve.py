
import sys
import matplotlib.pyplot as plt

import numpy
sys.path.append('/home/pesong/tools/ssd-caffe/python')
import caffe
from utils import score, surgery
import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass


# train from fine tune
weights = 'pretrained/mobilenet_iter_73000.caffemodel'
proto = 'pretrained/MobileNetSSD_train.prototxt'

# train from pretrained ssd with bdd100k
# weights = 'snapshot/seg/300/_iter_2000.caffemodel'
# proto = 'proto/seg/MobileNetSSD_train.prototxt'

final_model_name = 'seg'
n_steps = 20000

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.get_solver('seg_solver_train.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/dl/data/cityscapes/cityscapes_ncs/val_test.txt', dtype=str)


for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
