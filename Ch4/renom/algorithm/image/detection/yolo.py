#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from renom.core import Node, to_value
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu


def build_truth(y, total_w, total_h, cells, classes):
    """Use to transform a list of objects per image into a image*cells*cells*(5+classes) matrix.
    Each cell in image can only be labeled for 1 object.

    "5" represents: objectness (0 or 1) and X Y W H

    ex:
    Input: 2 objects in first image, 5 classes

    y[0] = X Y W H 0 1 0 0 0 X Y W H 0 0 0 1 0
          |---1st object----||---2nd object---|

    Output: 7 * 7 cells * 10 per image

    | truth[0,0,0] = 1 X Y W H 0 1 0 0
    | (cell 0,0 has first object)

    | truth[0,0,1] = 0 0 0 0 0 0 0 0 0
    | (cell 0,1 has no object)
    """
    truth = np.zeros((y.shape[0], cells, cells, 5 + classes))
    for im in range(y.shape[0]):
        for obj in range(0, y.shape[1], 4 + classes):
            truth_classes = y[im, obj + 4:obj + 4 + classes]
            if np.all(truth_classes == 0):
                continue
            truth_x = y[im, obj]
            truth_y = y[im, obj + 1]
            truth_w = y[im, obj + 2]
            truth_h = y[im, obj + 3]
            norm_x = truth_x * .99 * cells / total_w
            norm_y = truth_y * .99 * cells / total_h
            norm_w = truth_w / total_w
            norm_h = truth_h / total_h
            truth[im, int(norm_y), int(norm_x)] = np.concatenate(
                ([1, norm_x % 1, norm_y % 1, norm_w, norm_h], truth_classes))
    truth = truth.reshape(y.shape[0], -1)
    return truth


def box_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    xA = np.fmax(b1_x1, b2_x1)
    yA = np.fmax(b1_y1, b2_y1)
    xB = np.fmin(b1_x2, b2_x2)
    yB = np.fmin(b1_y2, b2_y2)
    intersect = (xB - xA) * (yB - yA)
    # case we are given two scalar boxes:
    if intersect.shape == ():
        if (xB < xA) or (yB < yA):
            return 0
    # case we are given an array of boxes:
    else:
        intersect[xB < xA] = 0.0
        intersect[yB < yA] = 0.0
    # 0.0001 to avoid dividing by zero
    union = (area1 + area2 - intersect + 0.0001)
    return intersect / union


def make_box(box):
    x1 = box[:, :, :, 0] - box[:, :, :, 2] / 2
    y1 = box[:, :, :, 1] - box[:, :, :, 3] / 2
    x2 = box[:, :, :, 0] + box[:, :, :, 2] / 2
    y2 = box[:, :, :, 1] + box[:, :, :, 3] / 2
    return [x1, y1, x2, y2]


def apply_nms(x, cells, bbox, classes, image_size, thresh=0.2, iou_thresh=0.3):
    u"""Apply to X predicted out of yolo_detector layer to get list of detected objects.
    Default threshold for detection is prob < 0.2.
    Default threshold for suppression is IOU > 0.4

    Args:
        cells (int): Cell size.
        bbox (int): Number of bbox.
        classes (int): Number of class.
        image_size (tuple): Image size.
        thresh (float): A threshold for effective bounding box.
        iou_thresh (float): A threshold for bounding box suppression.

    Returns:
        List of dict object is returned. The dict includes keys ``class``,
            ``box``, ``score``.
    """
    probs = np.zeros((cells, cells, bbox, classes))
    boxes = np.zeros((cells, cells, bbox, 4))  # 4 is x y w h
    for b in range(bbox):
        # bbox prob is "confidence of a bbox * probability of a class"
        prob = x[:, :, b * 5] * x[:, :, bbox * 5:].transpose(2, 0, 1)
        probs[:, :, b, :] = prob.transpose(1, 2, 0)
        boxes[:, :, b, :] = x[:, :, b * 5 + 1:b * 5 + 5]
        boxes[:, :, b, 2] = boxes[:, :, b, 2]
        boxes[:, :, b, 3] = boxes[:, :, b, 3]
    offset = np.array([np.arange(cells)] * (cells * bbox)
                      ).reshape(bbox, cells, cells).transpose(1, 2, 0)
    # offset x and y values to be 0<xy<1 for the whole image, not a cell
    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += offset.transpose(1, 0, 2)
    boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / float(cells)
    # get a single list of all bbox and pred for this image
    probs = probs.reshape(-1, classes)
    boxes = boxes.reshape(-1, 4)  # 4 is x y w h
    # filter bbox with prob less than thresh (default 0.2)
    probs[probs < thresh] = 0
    # reorder results by highest prob
    argsort = np.argsort(probs, axis=0)[::-1]
    # perform nms for boxes of same class

    def get_xy12(box):
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2
        return [x1, y1, x2, y2]
    for cl in range(classes):
        for b in range(boxes.shape[0]):
            if probs[argsort[b, cl], cl] == 0:
                continue
            b1 = boxes[argsort[b, cl], :]
            b1 = get_xy12(b1)
            for compar in range(b + 1, boxes.shape[0]):
                b2 = boxes[argsort[compar, cl], :]
                b2 = get_xy12(b2)
                if box_iou(b1, b2) > iou_thresh:
                    probs[argsort[compar, cl], cl] = 0
    classes = np.argmax(probs, axis=1)
    # filter only remaining boxes
    filter_iou = probs > 0
    indexes = np.nonzero(filter_iou)
    # save results
    results = []
    for b in range(len(indexes[0])):
        result = {
            "class": indexes[1][b],
            "box": boxes[indexes[0][b]],
            "score": probs[indexes[0][b], indexes[1][b]]
        }
        results.append(result)
    return results


class yolo(Node):

    def __new__(cls, x, y, cells, bbox, classes):
        return cls.calc_value(x, y, cells, bbox, classes)

    @classmethod
    def _oper_cpu(cls, x, y, cells, bbox, classes):
        x.to_cpu()
        noobj_scale = 0.5
        obj_scale = 5
        N = x.shape[0]  # np.prod(x.shape)
        raw_x = x
        x = x.as_ndarray()
        x = x.reshape(-1, cells, cells, (5 * bbox) + classes)
        y = y.reshape(-1, cells, cells, 5 + classes)
        deltas = np.zeros_like(x)
        loss = 0
        # Case: there's no object in the cell
        bg_ind = (y[:, :, :, 0] == 0)
        # Case: there's an object
        obj_ind = (y[:, :, :, 0] == 1)
        # add 5th part of the equation
        deltas[obj_ind, bbox * 5:] = x[obj_ind, bbox * 5:] - y[obj_ind, 5:]
        loss += np.sum(np.square(x[obj_ind, bbox * 5:] - y[obj_ind, 5:]))
        # search for the best predicted bounding box
        truth_box = make_box(y[:, :, :, 1:5])
        ious = np.zeros((y.shape[0], y.shape[1], y.shape[2], bbox))

        for b in range(bbox):
            # add 4th part of the equation
            deltas[bg_ind, b * 5] = noobj_scale * x[bg_ind, b * 5]
            loss += noobj_scale * np.sum(np.square(x[bg_ind, b * 5]))
            # get ious for current box
            box = x[:, :, :, 5 * b + 1:5 * b + 5]
            predict_box = make_box(box)
            ious[:, :, :, b] = box_iou(truth_box, predict_box)
        best_ind = np.argmax(ious, axis=3)

        for b in range(bbox):
            update_ind = (b == best_ind) & obj_ind
            # add 3rd part of the equation
            loss += np.sum(np.square(x[update_ind, 5 * b] - 1))
            deltas[update_ind, 5 * b] = (x[update_ind, 5 * b] - 1)

            # add 1st-2nd part of the equation
            loss += obj_scale * \
                np.sum(np.square(x[update_ind, 5 * b + 1:5 * b + 5] - y[update_ind, 1:5]))

            deltas[update_ind, 5 * b + 1:5 * b + 5] = obj_scale * \
                (x[update_ind, 5 * b + 1:5 * b + 5] - y[update_ind, 1:5])

        loss = loss / 2 / N
        deltas = deltas.reshape(-1, cells * cells * (5 * bbox + classes)) / N
        ret = cls._create_node(loss)
        ret.attrs._x = raw_x
        ret.attrs._deltas = deltas
        return ret

    @classmethod
    def _oper_gpu(cls, x, y, cells, bbox, classes):
        return cls._oper_cpu(x, y, cells, bbox, classes)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, self.attrs._deltas * dy)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, get_gpu(self.attrs._deltas) * dy)


class Yolo(object):
    u"""Loss function for Yolo detection.
    Last layer of the network needs to be following size:
    cells*cells*(bbox*5+classes)
    5 is because every bounding box gets 1 score and 4 locations (x, y, w, h)

    Ex:
    Prediction: 2 bbox per cell, 7*7 cells per image, 5 classes
    X[0,0,0] = S  X  Y  W  H  S  X  Y  W  H  0 0 0 1 0
              |---1st bbox--||---2nd bbox--||-classes-|

    Args:
        cells (int): Number of grid cells.
        bbox (int): Number of bbox.
        classes (int): Number of class.
    """

    def __init__(self, cells=7, bbox=2, classes=10):
        self._cells = cells
        self._bbox = bbox
        self._classes = classes

    def __call__(self, x, y):
        return yolo(x, y, self._cells, self._bbox, self._classes)
