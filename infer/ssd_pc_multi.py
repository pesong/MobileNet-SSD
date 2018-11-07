import numpy as np
import sys, os
import cv2

sys.path.append('/home/pesong/tools/ssd-caffe/python')
import caffe


# use GPU
# caffe.set_device(0)
# caffe.set_mode_gpu()

# param
mobile_type = "ssd"
W = 300
H = 300

net_file = '../proto/{}/MobileNetSSD_deploy_multi.prototxt'.format(mobile_type)
caffe_model = '../proto/{}/MobileNetSSD_deploy_multi.caffemodel'.format(mobile_type)
# test_dir = "/media/pesong/e/dl_gaussian/data/000/"
test_dir = "/dl/model/MobileNet-SSD/images/gs"


if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file, caffe_model, caffe.TEST)


# CLASSES = ('background',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')


CLASSES = ('background', 'person',  'car', 'bike', 'bus', 'rider')
out_len = 5 + 2 * len(CLASSES)

def preprocess(src):
    img = cv2.resize(src, (H, W))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out[0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out[0, 0, :, 1]
    conf = out[0, 0, :, 2]
    score_others = out[0, 0, :, 7:out_len]
    return (box.astype(np.int32), conf, cls, score_others)


def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()['detection_out']
    box, conf, cls, score_others = postprocess(origimg, out)

    print("-----------------------------------------")
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(origimg, p1, p2, (0, 255, 0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.3f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.4, (0, 255, 0), 1)

        print("bbox ", i)
        print(CLASSES[int(cls[i])],  ": ", ("%.3f" %conf[i]) )
        for j in range(len(CLASSES) - 1):
            print(CLASSES[int(score_others[i][2*j])], ": ", ("%.3f" % score_others[i][2*j+1]))
        print('\n')

    cv2.imshow("SSD", origimg)

    k = cv2.waitKey(0) & 0xff
    # Exit if ESC pressed
    if k == 27: return False
    return True


for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
        break
