import os
import cv2
import numpy as np

root_path = '/media/pesong/e/dl_gaussian/data/backbone_seg_ssd/cityscapes_ssd_seg/Segmentations'
dst_path = '/media/pesong/e/dl_gaussian/data/backbone_seg_ssd/cityscapes_ssd_seg/Segmentations'

for file in os.listdir(root_path):
    file_name = os.path.join(root_path, file)
    # img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(file_name)
    a_img = np.array(img, np.double)
    normalized = cv2.normalize(img, a_img, 1.0, 0.0, cv2.NORM_MINMAX)
    dst_file = os.path.join(dst_path, file)
    cv2.imwrite(dst_file, normalized)
