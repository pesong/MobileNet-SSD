#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
import os
import time

import numpy
import cv2
import sys
import mvnc.mvncapi as mvnc
from utils import vis

IMAGE_PATH_ROOT = '/dl/model/MobileNet-SSD/images/gs/'
W = 300
H = 300

# ***************************************************************
# Labels for the classifications for the network.
# ***************************************************************
# LABELS = ('background',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

LABELS = ('background', 'person',  'car', 'bike', 'bus', 'rider')

# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_ssd_inference(input_fifo, output_fifo, input_tensor, image_to_classify, ssd_mobilenet_graph):

    # Write the tensor to the input_fifo and queue an inference
    ssd_mobilenet_graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, input_tensor, 'user object')

    # ***************************************************************
    # Get the results from the output queue
    output, user_obj = output_fifo.read_elem()

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))

    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
            x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[0]))
            y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[1]))
            x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[0]))
            y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[1]))

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')

            # overlay boxes and labels on the original image to classify
            overlay_on_image(image_to_classify, output[base_index:base_index + 7])


# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info):

    # the minimal score for a box to be shown
    min_score_percent = 60

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):

    # scale the image
    NETWORK_WIDTH = W
    NETWORK_HEIGHT = H
    resized_img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))


    # adjust values to range between -1.0 and + 1.0
    input_img = resized_img - 127.5
    input_img = input_img * 0.007843
    return resized_img, input_img


# This function is called from the entry point to do
# all the work of the program
def main():

    # look for device
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print("No devices found")
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.open()

    # The graph file that was created with the ncsdk compiler
    graph_file_name_ssd = '/dl/model/MobileNet-SSD/proto/ssd/MobileNetSSD_deploy.graph'

    # Load graph file data
    with open(graph_file_name_ssd, mode='rb') as f_ssd:
        graph_from_disk_ssd = f_ssd.read()

    # Initialize a Graph object
    graph_ssd = mvnc.Graph('graph_ssd')

    # Allocate the graph to the device and create input and output Fifos with default arguments
    input_fifo_ssd, output_fifo_ssd = graph_ssd.allocate_with_fifos(device, graph_from_disk_ssd)

    i = 0
    start = time.time()
    for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):

        # read the image to run an inference on from the disk
        IMAGE_FULL_PATH = os.path.join(IMAGE_PATH_ROOT, IMAGE_PATH)
        image_read = cv2.imread(IMAGE_FULL_PATH)
        # b, g, r = cv2.split(image_read)
        # image_read = cv2.merge([r, g, b])

        # get a resized version of the image that is the dimensions
        # SSD Mobile net expects
        resized_image, input_image = preprocess_image(image_read)

        # Convert an input tensor to 32FP data type
        Image_tensor = input_image.astype(numpy.float32)

        # run a single inference on the image and overwrite the
        # boxes and labels
        run_ssd_inference(input_fifo_ssd, output_fifo_ssd, Image_tensor, resized_image, graph_ssd)

        # display the results and wait for user to hit a key
        cv2.imshow("ssd_out_image", resized_image)
        cv2.waitKey(0)

        i += 1
        duration = time.time() - start
        floaps = i / duration
        print("time:{}, images_num:{}, floaps:{}".format(duration, i, floaps))


    # Clean up
    input_fifo_ssd.destroy()
    output_fifo_ssd.destroy()
    graph_ssd.destroy()

    device.close()
    device.destroy()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())