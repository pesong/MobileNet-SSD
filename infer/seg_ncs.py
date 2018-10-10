#! /usr/bin/env python3

import os
import time
import numpy
import skimage.io
import skimage.transform
from PIL import Image
import mvnc.mvncapi as mvnc
from utils import vis
import matplotlib.pyplot as plt


# input parameters
IMAGE_MEAN = [127.5, 127.5, 127.5]

graph_file_name = '/dl/model/MobileNet-SSD/proto/seg/MobileNetSSD_deploy.graph'
IMAGE_PATH_ROOT = '/dl/model/MobileNet-SSD/images/CS/'
IMAGE_DIM = [320, 480]


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
fig = plt.figure(figsize=(18,12))
fig.tight_layout()
plt.subplots_adjust(left=0.04, top= 0.96, right = 0.96, bottom = 0.04, wspace = 0.01, hspace = 0.01)
plt.ion()

i = 0
start = time.time()
for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):

    img_ori = skimage.io.imread(os.path.join(IMAGE_PATH_ROOT + IMAGE_PATH))

    # Resize image [Image size is defined during training]
    img = skimage.transform.resize(img_ori, IMAGE_DIM, preserve_range=True)

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype(numpy.float32)
    image_t = (img - numpy.float32(IMAGE_MEAN))
    # image_t = numpy.transpose(image_t, (2, 0, 1))

# -----------step4: get result-------------------------------------------------
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, image_t, 'user object')

    # Get the results from NCS
    out, userobj = fifo_out.read_elem()

    #  flatten ---> image
    out = out.reshape(-1, 2).T.reshape(2, 331, -1)
    out = out.argmax(axis=0)
    out = out[6:-5, 6:-5]

    # save result
    voc_palette = vis.make_palette(2)
    out_im = Image.fromarray(vis.color_seg(out, voc_palette))
    iamge_name = IMAGE_PATH.split('/')[-1].rstrip('.jpg')
    # out_im.save('demo_test/' + iamge_name + '_ncs_' + '.png')

    # get masked image
    img_masked = Image.fromarray(vis.vis_seg(img_ori, out, voc_palette))
    # masked_im.save('demo_test/visualization.jpg')

    i += 1
    duration = time.time() - start
    floaps = i / duration
    print("time:{}, images_num:{}, floaps:{}".format(duration, i, floaps))


    # draw picture
    plt.suptitle('MobilenetV1-movidius', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.title("orig image", fontsize=16)
    plt.imshow(img_ori)

    plt.subplot(1, 2, 2)
    plt.title("segmentation", fontsize=16)
    plt.imshow(img_masked)

    plt.pause(0.000001)
    plt.clf()

plt.ioff()
plt.show()

# Clean up the graph, device, and fifos
fifo_in.destroy()
fifo_out.destroy()
graph.destroy()
device.close()
device.destroy()
