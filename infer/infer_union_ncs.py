#! /usr/bin/env python3

import os
import time
import numpy as np
import skimage.io
import skimage.transform
from PIL import Image
import mvnc.mvncapi as mvnc
from utils import vis
import cv2
import matplotlib.pyplot as plt


# input parameters
H = 320
W = 480
CLS = 2
IMAGE_DIM = [H, W]
IMAGE_MEAN = [127.5, 127.5, 127.5]

CLASSES = ('background',
           'person', 'car')


graph_file_name = '/dl/model/MobileNet-SSD/proto/union/union.graph'
IMAGE_PATH_ROOT = '/dl/model/MobileNet-SSD/images/CS/'


# function used by ssd infer

def preprocess(src):
    img = cv2.resize(src, (W, H))
    img = img - 127.5
    img = img * 0.007843
    img = img.astype(np.float32)
    img = img.transpose((CLS, 0, 1))
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out[0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out[0, 0, :, 1]
    conf = out[0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)


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


i = 0
start = time.time()
for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):

    img_ori = skimage.io.imread(os.path.join(IMAGE_PATH_ROOT + IMAGE_PATH))

    # Resize image [Image size is defined during training]
    img = skimage.transform.resize(img_ori, IMAGE_DIM, preserve_range=True)

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype(np.float32)
    image_t = (img - np.float32(IMAGE_MEAN)) * np.float32(2.0/255)
    # image_t = numpy.transpose(image_t, (2, 0, 1))

# ----------- step3 : get result-------------------------------------------------
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, image_t, 'user object')

    # Get the results from NCS
    out, userobj = fifo_out.read_elem()
    total_out = out['total_out'].reshape([1, -1])[0]

    # out_ssd
    out_ssd = total_out[H * W * CLS:].reshape([1, 1, -1, 7])
    box, conf, cls = postprocess(img_ori, out_ssd)


    # out_seg:  flatten ---> image
    out_seg = total_out[0: H * W * CLS]
    out_seg = out_seg.reshape([1, CLS, H, W])
    out_seg = out_seg[0].argmax(axis=0)

    # -------------visualize segmentation------------------
    voc_palette = vis.make_palette(CLS)  # CLS代表分割模型的类别数目
    out_im = Image.fromarray(vis.color_seg(out_seg, voc_palette))
    # iamge_name = img_path.split('/')[-1].rstrip('.jpg')
    # out_im.save('demo_test/' + iamge_name + '_pc_' + '.png')

    # 对原始照片融合mask像素信息
    img_masked_array = vis.vis_seg(img_ori, out_seg, voc_palette)
    img_masked = Image.fromarray(img_masked_array)

    # img_masked.save('demo_test/visualization.jpg')

    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(img_masked_array, p1, p2, (0, 255, 0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(img_masked_array, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

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
