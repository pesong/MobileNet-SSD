#!/bin/sh
#latest=snapshot/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/ssd/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
/home/pesong/tools/ssd-caffe/build/tools/caffe train -solver="solver_test_ssd.prototxt" \
--weights=$latest \
-gpu 0
