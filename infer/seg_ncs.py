#! /usr/bin/env python3

import os
import time

import numpy as np
from PIL import Image
import mvnc.mvncapi as mvnc
from utils import vis
import cv2


# input parameters
IMAGE_MEAN = [127.5, 127.5, 127.5]

graph_file_name = '/dl/model/MobileNet-SSD/proto/seg/MobileNetSSD_deploy.graph'
IMAGE_PATH_ROOT = '/dl/model/MobileNet-SSD/images/CS/'
# IMAGE_PATH_ROOT = '/media/pesong/e/dl_gaussian/data/000/'

IMAGE_DIM = (300, 300)


def preprocess(src):
    img = src - 127.5
    img = img * 0.007843
    img = img.astype(np.float32)
    # img = img.transpose((2, 0, 1))
    return img


# configure the NCS
# ***************************************************************
mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

# --------step1: open the device and get a handle to it--------------------
# look for device
devices = mvnc.enumerate_devices()
if len(devices) == 0:
    print("No devices found")
    quit()

# Pick the first stick to run the network
device = mvnc.Device(devices[0])

# Open the NCS
device.open()


# ---------step2: load a graph file into hte ncs device----------------------
# Load network graph file into memory
with open(graph_file_name, mode='rb') as f:
    blob = f.read()

# create and allocate the graph object
graph = mvnc.Graph('graph')
fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

# -------- step3: offload image into the ncs to run inference


i = 0
start = time.time()
for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):

    img_ori = cv2.imread(os.path.join(IMAGE_PATH_ROOT, IMAGE_PATH))
    b, g, r = cv2.split(img_ori)
    img_ori = cv2.merge([r, g, b])

    img_resize = cv2.resize(img_ori, IMAGE_DIM)
    img_in = preprocess(img_resize)


# -----------step4: get result-------------------------------------------------
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img_in, 'user object')

    # Get the results from NCS
    out, userobj = fifo_out.read_elem()

    #  flatten ---> image
    out = out.reshape(-1, 2).T.reshape(2, 300, -1)
    out_seg = out.argmax(axis=0)
    # out = out[6:-5, 6:-5]

    # save result
    voc_palette = vis.make_palette(2)  # 2代表分割模型的类别数目
    out_im = Image.fromarray(vis.color_seg(out_seg, voc_palette))
    iamge_name = IMAGE_PATH.split('/')[-1].rstrip('.jpg')
    # out_im.save('demo_test/' + iamge_name + '_ncs_' + '.png')


    # 对原始照片融合mask像素信息
    img_masked_array = vis.vis_seg(img_resize, out_seg, voc_palette)
    img_masked = Image.fromarray(img_masked_array)
    img_masked_array = img_masked_array[:, :, ::-1]
    cv2.imshow("SSD", img_masked_array)
    # k = cv2.waitKey(1)

    # # Exit if ESC pressed
    k = cv2.waitKey(0) & 0xff
    if k == 27: break

    i += 1
    duration = time.time() - start
    floaps = i / duration
    print("time:{}, images_num:{}, floaps:{}".format(duration, i, floaps))




# Clean up the graph, device, and fifos
fifo_in.destroy()
fifo_out.destroy()
graph.destroy()
device.close()
device.destroy()
