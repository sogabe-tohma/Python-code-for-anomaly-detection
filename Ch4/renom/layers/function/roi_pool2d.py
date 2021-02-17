#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from renom.core import Node, to_value
from renom.layers.function.utils import roi_pooling_slice, region_cordinates, roi_pooling_slice_decode
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu


class roi_pool2d(Node):

    def __new__(cls, x, rois, outh=7, outw=7, spatial_scale=1 / 16.):
        ch, h, w = x.shape[1:]
        n_rois = rois.shape[0]
        return cls.calc_value(x, rois, ch, h, w, n_rois, outh, outw, spatial_scale=1 / 16.)

    @classmethod
    def _oper_cpu(cls, x, rois, ch, h, w, n_rois, outh, outw, spatial_scale):
        z = np.zeros((n_rois, ch, outh, outw), np.float64)
        index = np.zeros(z.shape, np.int32)

        for i_roi in range(n_rois):
            idx, xmin, ymin, xmax, ymax = region_cordinates(rois[i_roi], spatial_scale)
            roi_height = max(ymax - ymin + 1, 1)
            roi_width = max(xmax - xmin + 1, 1)
            strideh = float(roi_height) / float(outh)
            stridew = float(roi_width) / float(outw)

            for idx_h in range(outh):
                sliceh, lenh = roi_pooling_slice(idx_h, strideh, h, ymin)
                if lenh <= 0:
                    continue
                for idx_w in range(outw):
                    slicew, lenw = roi_pooling_slice(idx_w, stridew, w, xmin)
                    if lenw <= 0:
                        continue
                    roi_data = x[int(idx), :, sliceh, slicew].reshape(ch, -1)
                    z[i_roi, :, idx_h, idx_w] = np.max(roi_data, axis=1)
                    max_idx_slice = np.unravel_index(np.argmax(roi_data, axis=1), (lenh, lenw))
                    max_idx_slice_h = max_idx_slice[0] + sliceh.start
                    max_idx_slice_w = max_idx_slice[1] + slicew.start
                    max_idx_slice = max_idx_slice_h * w + max_idx_slice_w
                    index[i_roi, :, idx_h, idx_w] = max_idx_slice
        ret = cls._create_node(z)
        ret.attrs._index = index
        ret.attrs._x = x
        ret.attrs._rois = rois
        ret.attrs._outw = outw
        ret.attrs._outh = outh
        ret.attrs._spatial_scale = spatial_scale
        return ret

    @classmethod
    def _oper_gpu(cls, x, rois, ch, h, w, n_rois, outh, outw, spatial_scale):
        z = GPUValue(shape=(n_rois, ch, outh, outw))
        argmax_data = z.empty_like_me()
        rois = get_gpu(rois)
        cu.curoi_pool2d_forward(rois, get_gpu(x), spatial_scale, ch,
                                h, w, outh, outw, z, argmax_data)
        ret = cls._create_node(z)
        ret.attrs._index = argmax_data
        ret.attrs._x = x
        ret.attrs._rois = rois
        ret.attrs._outh = outh
        ret.attrs._outw = outw
        ret.attrs._spatial_scale = spatial_scale
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            ch, h, w = self.attrs._x.shape[1:]
            n_rois = self.attrs._rois.shape[0]
            dx = np.zeros_like(self.attrs._x)

            for i_roi in range(n_rois):
                idx, xmin, ymin, xmax, ymax = region_cordinates(self.attrs._rois[i_roi],
                                                                self.attrs._spatial_scale)
                roi_height = max(ymax - ymin + 1, 1)
                roi_width = max(xmax - xmin + 1, 1)

                stride_h = float(roi_height) / float(self.attrs._outh)
                stride_w = float(roi_width) / float(self.attrs._outw)
                for idx_h in range(ymin, ymax + 1):
                    for idx_w in range(xmin, xmax + 1):
                        start_w, end_w = roi_pooling_slice_decode(
                            idx_w, stride_w, self.attrs._outw, xmin)
                        start_h, end_h = roi_pooling_slice_decode(
                            idx_h, stride_h, self.attrs._outh, ymin)

                        for ph in range(start_h, end_h):
                            for pw in range(start_w, end_w):
                                max_idx_tmp = self.attrs._index[i_roi, :, ph, pw]
                                for c in range(ch):
                                    if max_idx_tmp[c] == (idx_h * w + idx_w):
                                        dx[idx, c, idx_h, idx_w] += dy[i_roi, c, ph, pw]

        self.attrs._x._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            ch, h, w = self.attrs._x.shape[1:]
            dx = GPUValue(shape=self.attrs._x.shape)
            cu.curoi_pool2d_backward(get_gpu(dy), self.attrs._index, self.attrs._rois,
                                     self.attrs._spatial_scale, ch, h, w, self.attrs._outh, self.attrs._outw, dx)
            self.attrs._x._update_diff(context, dx, **kwargs)


class RoiPoolBase(object):
    def __init__(self, outh=7, outw=7, spatial_scale=1 / 16.):
        self.outw = outw
        self.outh = outh
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois):
        return self.forward(x, rois)


class RoiPool2d(RoiPoolBase):
    def forward(self, x, rois):
        return roi_pool2d(x, rois, self.outh, self.outw, self.spatial_scale)
