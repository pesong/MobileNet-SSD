import os
import sys
import time

caffe_root = '/home/pesong/tools/ssd-caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import vis

caffe.set_device(0)
caffe.set_mode_gpu()

# define parameters
img_path_root = '/dl/model/MobileNet-SSD/images/gs'
# img_path_root = '/media/pesong/e/dl_gaussian/data/000/'

IMAGE_MEAN = [127.5, 127.5, 127.5]
IMAGE_DIM = (480, 320)

NET_PROTO = "/dl/model/MobileNet-SSD/proto/union/MobileNetSSD_deploy.prototxt"
WEIGHTS = '/dl/model/MobileNet-SSD/proto/union/MobileNetSSD_deploy.caffemodel'

CLASSES = ('background', 'person',  'car', 'bike', 'bus', 'rider')


def preprocess(src):

    img = src - 127.5
    img = img * 0.007843
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out[0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out[0, 0, :, 1]
    conf = out[0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)


# load net
net = caffe.Net(NET_PROTO, WEIGHTS, caffe.TEST)

processed_img_num = 0
start = time.time()
for img_path in os.listdir(img_path_root):

    img_ori = cv2.imread(os.path.join(img_path_root, img_path))
    b, g, r = cv2.split(img_ori)
    img_ori = cv2.merge([r, g, b])

    img_resize = cv2.resize(img_ori, IMAGE_DIM)
    img_pre = preprocess(img_resize)

    # # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = img_pre

    # ------------------infer-----------------------------
    # run net and take argmax for prediction
    out = net.forward()
    out_ssd = out['detection_out']
    box, conf, cls = postprocess(img_resize, out_ssd)

    out_seg = out['score']
    out_seg = out_seg[0].argmax(axis=0)

    # -------------visualize segmentation------------------
    voc_palette = vis.make_palette(2)  # 2代表分割模型的类别数目
    out_im = Image.fromarray(vis.color_seg(out_seg, voc_palette))
    # iamge_name = img_path.split('/')[-1].rstrip('.jpg')
    # out_im.save('demo_test/' + iamge_name + '_pc_' + '.png')

    # 对原始照片融合mask像素信息
    img_masked_array = vis.vis_seg(img_resize, out_seg, voc_palette)
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

    cv2.namedWindow("union", 0)
    cv2.resizeWindow("union", 600, 400)
    cv2.imshow("union", img_masked_array)
    # k = cv2.waitKey(1)

    # # Exit if ESC pressed
    k = cv2.waitKey(0) & 0xff
    if k == 27: break

    # 统计推理速度
    processed_img_num += 1
    duration = time.time() - start
    floaps = processed_img_num / duration
    print("time:{}, images_num:{}, floaps:{}".format(duration, processed_img_num, floaps))


