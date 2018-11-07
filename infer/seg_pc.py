import os
import sys
import time
caffe_root = '/home/pesong/tools/ssd-caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2

import numpy as np
from utils import vis

caffe.set_device(0)
caffe.set_mode_gpu()

# define parameters
img_path_root = '/dl/model/MobileNet-SSD/images/CS'
# img_path_root ='/media/pesong/e/dl_gaussian/data/000/'

IMAGE_MEAN = [127.5, 127.5, 127.5]
IMAGE_DIM = (300, 300)

NET_PROTO = "/dl/model/MobileNet-SSD/proto/seg/MobileNetSSD_deploy.prototxt"
WEIGHTS = '/dl/model/MobileNet-SSD/proto/seg/MobileNetSSD_deploy.caffemodel'


def preprocess (src):
    img = src - 127.5
    img = img * 0.007843
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    return img

# load net
net = caffe.Net(NET_PROTO, WEIGHTS, caffe.TEST)

i = 0
start = time.time()
for img_path in os.listdir(img_path_root):

    img_ori = cv2.imread(os.path.join(img_path_root, img_path))
    b, g, r = cv2.split(img_ori)
    img_ori = cv2.merge([r, g, b])

    img_resize = cv2.resize(img_ori, IMAGE_DIM)
    img_in = preprocess(img_resize)

    # # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = img_in

    # ------------------infer-----------------------------
    # run net and take argmax for prediction
    net.forward()
    out_seg = net.blobs['score'].data[0]
    out_seg = out_seg.argmax(axis=0)

    # -------------visualize segmentation------------------
    voc_palette = vis.make_palette(2)  # 2代表分割模型的类别数目

    # 对原始照片融合mask像素信息
    img_masked_array = vis.vis_seg(img_resize, out_seg, voc_palette)

    img_masked_array = img_masked_array[:, :, ::-1]
    cv2.imshow("SSD", img_masked_array)
    # k = cv2.waitKey(1)

    # # Exit if ESC pressed
    k = cv2.waitKey(0) & 0xff
    if k == 27: break

