#!/bin/sh
#latest=snapshot/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/union/cp/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
/home/pesong/tools/ssd-caffe/build/tools/caffe train -solver="ssd_solver_test.prototxt" \
--weights=$latest \
-gpu 0
