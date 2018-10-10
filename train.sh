#!/bin/sh
if ! test -f proto/ssd/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot/ssd
/home/pesong/tools/ssd-caffe/build/tools/caffe train -solver="solver_train_ssd.prototxt" \
-weights="pretrained/mobilenet_iter_73000.caffemodel" \
-gpu 0 
