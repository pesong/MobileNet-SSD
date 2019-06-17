'''
this script genetate the data include the
ori jpegimages
detection label
segmentation label png
'''

import os
import shutil


ori_path = "/dl/data/VOCdevkit/VOC2012"
dst_path = "/dl/data/backbone_seg_ssd/voc_seg"

if not os.path.exists(dst_path):
    os.makedirs(os.path.join(dst_path, "JPEGImages"))
    os.makedirs(os.path.join(dst_path, "ImageSets"))
    os.makedirs(os.path.join(dst_path, "Annotations"))



for fname in ["trainval.txt" , "test.txt"]:
    f = open(fname, 'r')
    for line in f.readlines():
        line = line.rstrip('\n')

        f_path_src = os.path.join(ori_path, "JPEGImages", line + ".jpg")
        f_path_dst = os.path.join(dst_path, "JPEGImages", line + ".jpg")
        shutil.copyfile(f_path_src, f_path_dst)

        label_path_src = os.path.join(ori_path, "Annotations", line + ".xml")
        label_path_dst = os.path.join(dst_path, "Annotations", line + ".xml")
        shutil.copyfile(label_path_src, label_path_dst)


shutil.copytree(os.path.join(ori_path, "SegmentationClass"), os.path.join(dst_path, "Segmentations"))
