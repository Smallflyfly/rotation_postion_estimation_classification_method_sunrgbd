#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# Edited by Matthew Seals
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse
from matplotlib import cm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__',
           'chair', 'table', 'sofa', 'bed', 'shelf', 'cabinet')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',), 'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

COLORS = [cm.tab10(i) for i in np.linspace(0., 1., 10)]

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'paper_demo', image_name)
    # print(im_file)
    im = cv2.imread(im_file)
    # print(im)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # scores, boxes, points2d = im_detect(net, im)
    scores, boxes, grid_scores, rotation_scores = im_detect(net, im)
    # print(np.max(front_4points))
    # print(np.max(back_4points))
    # print(np.max(center))
    # fang[-1]
    # scores, boxes, center = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    thresh = 0.8  # CONF_THRESH
    NMS_THRESH = 0.3

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    cntr = -1
    # print(scores.shape

    grid_scores_det = grid_scores.max(1)
    inds_grid = grid_scores.argmax(1)
    rotation_scores_det = rotation_scores.max(1)
    inds_rotation = rotation_scores.argmax(1)
    # print(grid_scores_det)
    # print(inds_grid)
    # fang[-1]

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background

        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]

        # cls_grid_scores = grid_scores[:, cls_ind]

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]

        grid_pred = inds_grid[keep.numpy()]
        rotation_pred = inds_rotation[keep.numpy()]


        inds = np.where(dets[:, -1] >= thresh)[0]
        print('inds', inds)
        
        if len(inds) == 0:
            continue
        else:
            cntr += 1
        # fang[-1]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            grid = grid_pred[i]
            rotation = rotation_pred[i]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=COLORS[cntr % len(COLORS)], linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    # '{:s} {:.3f} {:d}'.format(cls, score, int(grid)),
                    '{:s} {:.3f} {:d} {:d}'.format(cls, score, int(grid), int(rotation)),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title('All detections with threshold >= {:.1f}'.format(thresh), fontsize=14)

        plt.axis('off')
        plt.tight_layout()
        plt.draw()
    plt.savefig('./paper_demo_out/demo_' + image_name)
    print('Saved to `{}`'.format(os.path.join(os.getcwd(), 'demo_' + image_name)))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    # saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
    #                            NETS[demonet][0] % (70000 if dataset == 'pascal_voc' else 110000))

    saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                            NETS[demonet][0] % (120000 if dataset == 'pascal_voc' else 110000))

    print(saved_model)                           

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(7, tag='default', anchor_scales=[8, 16, 32])  # class 7 #fang

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    im_names = [i for i in os.listdir('data/paper_demo/')  # Pull in all jpgs
                if i.lower().endswith(".jpg")]
    print(im_names)                
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/points2d_demo/{}'.format(im_name))
        demo(net, im_name)

    # plt.show()
