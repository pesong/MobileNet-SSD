import lmdb
import sys
caffe_root = '/home/pesong/tools/ssd-caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from matplotlib import pyplot
from caffe.proto import caffe_pb2
import cv2


lmdbpath = "seg_trainval_lmdb"
lmdb_env = lmdb.open(lmdbpath, readonly=True)

lmdb_txn = lmdb_env.begin()                                 # 生成处理句柄
lmdb_cursor = lmdb_txn.cursor()                             # 生成迭代器指针
annotated_datum = caffe_pb2.AnnotatedDatum()                # AnnotatedDatum结构

for key, value in lmdb_cursor:
    print(key)

    annotated_datum.ParseFromString(value)
    datum = annotated_datum.datum                           # Datum结构
    grps = annotated_datum.annotation_group                 # AnnotationGroup结构
    type = annotated_datum.type
    datum_label = annotated_datum.datum_label

    for grp in grps:

        for ann in grp.annotation:
            xmin = ann.bbox.xmin * datum.width           # Annotation结构
            ymin = ann.bbox.ymin * datum.height
            xmax = ann.bbox.xmax * datum.width
            ymax = ann.bbox.ymax * datum.height

            print ("label:", grp.group_label)                            # object的name标签
            print ("bbox:", xmin, ymin, xmax, ymax)                      # object的bbox标签

    label = datum_label.label                                      # Datum结构label以及三个维度
    channels = datum_label.channels
    height = datum_label.height
    width = datum_label.width
    encoded = datum_label.encoded
    is_seg = datum.is_seg


    print ("label:", label)
    print ("channels:", channels)
    print ("height:", height)
    print ("width:", width)
    print ("type:", type)
    print ("encoded:", encoded)
    print("is_seg:", is_seg)

    image_x = np.fromstring(datum.data, dtype=np.uint8)      # 字符串转换为矩阵
    image = cv2.imdecode(image_x, -1)

    image_y = np.fromstring(datum_label.data, dtype=np.uint8)  # 字符串转换为矩阵
    image_label = cv2.imdecode(image_y, -1)  # decode


    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.imshow("image", image_label)  # 显示图片
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


