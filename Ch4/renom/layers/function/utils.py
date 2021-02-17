# -*- coding: utf - 8 -*-
import numpy as np
from renom.core import to_value
from renom import precision


def out_size(size, k, s, p, d=(1, 1), ceil_mode=False):
    size = np.array(size)
    k = np.array(k)
    s = np.array(s)
    p = np.array(p)
    d = np.array(d)
    dk = k + (k - 1) * (d - 1)
    if ceil_mode:
        out = ((size + p * 2 - dk + s - 1) // s + 1).astype(np.int)
    else:
        out = ((size + p * 2 - dk) // s + 1).astype(np.int)
    return out


def transpose_out_size(size, k, s, p, d=(1, 1)):
    return (np.array(s) * (np.array(size) - 1) + np.array(k) + (np.array(k) - 1) *
            (np.array(d) - 1) - 2 * np.array(p)).astype(np.int)


def im2col(img, size, kernel, stride, padding, dilation=(1, 1), padWith=0.):
    N, channel, in_h, in_w = img.shape
    out_h, out_w = size
    k_h, k_w = kernel
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation
    img_n = np.pad(img, ((0, 0), (0, 0), (p_h, p_h + s_h - 1),
                         (p_w, p_w + s_w - 1)), mode="constant", constant_values=padWith)
    col = np.ndarray((N, channel, k_h, k_w, out_h, out_w), dtype=precision)
    for i in range(k_h):
        idh = i * d_h
        iu = idh + s_h * out_h

        for j in range(k_w):
            jdw = j * d_w
            ju = jdw + s_w * out_w
            col[:, :, k_h - 1 - i, k_w - 1 - j, :, :] = img_n[:, :, idh:iu:s_h, jdw:ju:s_w]
    return col


def pad_image(img, padding, stride, padWith=0.):
    dims = img.shape[2:]
    dimensionality = len(dims)
    pad_tuple = (padding, padding + stride - 1)
    pad_list = [(0, 0), (0, 0)]
    pad_list.extend([pad_tuple for _ in range(dimensionality)])
    padded_image = np.pad(img, tuple(pad_list),
                          mode="constant", constant_values=padWith)
    return padded_image


def imncol(img, weight, stride, padding, padWith=0.):
    N, in_channels, in_dims = img.shape[0], img.shape[1], img.shape[2:]
    out_channels = weight.shape[0]
    assert in_channels == weight.shape[1], "Number of feature maps is not the same for input and output"
    dimensionality = len(in_dims)

    # Padding asks for (before, after) for each dimension or it generalizes the padding
    pad_list = [(0, 0), (0, 0)]
    pad_list.extend([(padding[i], padding[i]) for i in range(dimensionality)])
    padded_image = np.pad(img, tuple(pad_list),
                          mode="constant", constant_values=padWith)
    ret = []
    for batch in range(N):
        tmp = []
        for out_channel in range(out_channels):
            tmp2 = 0
            for in_channel in range(in_channels):
                tmp2 += place_kernels(padded_image[batch, in_channel],
                                      weight[out_channel, in_channel], stride=stride)
            tmp.append(tmp2)
        ret.append(tmp)
    ret = np.array(ret)
    return np.array(ret)


def colnim(img, weight, stride):
    ret = []
    for batch in range(img.shape[0]):
        tmp2 = 0
        for out_channel in range(weight.shape[0]):
            tmp = []
            for in_channel in range(weight.shape[1]):
                tmp.append(place_back_kernels(img[batch, out_channel],
                                              weight[out_channel, in_channel], stride=stride))
            tmp2 += np.array(tmp)
        ret.append(tmp2)
    ret = np.array(ret)
    return ret


def colnw(img, weight, stride):
    ret = []
    for out_channel in range(weight.shape[1]):
        tmp2 = 0
        for batch in range(img.shape[0]):
            tmp = []
            for in_channel in range(img.shape[1]):
                tmp.append(place_overlap_kernels(
                    img[batch, in_channel], weight[batch, out_channel], stride=stride))
            tmp2 += np.array(tmp)
        ret.append(tmp2)
    ret = np.array(ret)
    return ret


def imnw(img, weight, stride):
    ret = []
    for out_channel in range(weight.shape[1]):
        tmp2 = 0
        for batch in range(img.shape[0]):
            tmp = []
            for in_channel in range(img.shape[1]):
                tmp.append(place_overlap_back_kernels(
                    img[batch, in_channel], weight[batch, out_channel], stride=stride))
            tmp2 += np.array(tmp)
        ret.append(tmp2)
    ret = np.array(ret)
    return ret


def place_kernels(img, kernel, stride, offset=0):
    kernels = []
    for i in range(len(img.shape)):
        kernels.append((img.shape[i] - kernel.shape[i] + offset * 2) // stride[i] + 1)
    kernels = np.zeros(tuple(kernels))
    assert len(kernel) > offset, "{}\{}".format(len(kernel), offset)
    for pos in generate_positions(img, stride, offset, min_space=np.array(kernel.shape) - 1):
        slices = tuple([slice(pos[i], pos[i] + kernel.shape[i]) for i in range(len(img.shape))])
        kern = np.sum(img[slices] * kernel)
        kernels[tuple(np.array(pos) // stride)] = kern
    return kernels


def place_back_kernels(img, kernel, stride=1, offset=0):
    ret_shape = (np.array(img.shape) - 1) * stride + np.array(kernel.shape)
    ret = np.zeros(ret_shape)
    itr_img = ret
    min = np.array(kernel.shape) - 1
    for pos in generate_positions(itr_img, stride, offset, min_space=min):
        slices = tuple([slice(pos[i], pos[i] + kernel.shape[i]) for i in range(len(img.shape))])
        kern = kernel * img[tuple(np.array(pos) // stride)]
        ret[slices] += kern
    return ret


def place_overlap_kernels(img, kernel, stride=1, offset=0):
    ret_shape = np.array(img.shape) - (np.array(kernel.shape) - 1) * stride
    ret = np.zeros(ret_shape)
    itr_img = img
    min = np.array(ret.shape) - 1
    for pos in generate_positions(itr_img, stride, offset, min_space=min):
        slices = tuple([slice(pos[i], pos[i] + ret.shape[i]) for i in range(len(img.shape))])
        kern = img[slices] * kernel[tuple(np.array(pos) // stride)]
        ret += kern
    return ret


def place_overlap_back_kernels(img, kernel, stride=1, offset=0):
    ret_shape = np.array(img.shape) - (np.array(kernel.shape) - 1) * stride
    ret = np.zeros(ret_shape)
    itr_img = img
    min = np.array(ret.shape) - 1
    for pos in generate_positions(itr_img, stride, offset, min_space=min):
        slices = [slice(pos[i], pos[i] + ret.shape[i]) for i in range(len(img.shape))]
        strided_slices = [slice(pos[i] // stride[i], pos[i] // stride[i] + ret.shape[i])
                          for i in range(len(img.shape))]
        kern = img[slices] * kernel[strided_slices]
        ret += kern
    return ret


def imnpool(img, kernel, stride, padding, padWith=0, mode="max", alternate_input=None):
    if mode is "max":
        func = max_pool
    elif mode is "average":
        func = average_pool

    N, in_channels, in_dims = img.shape[0], img.shape[1], img.shape[2:]
    dimensionality = len(in_dims)
    pad_list = [(0, 0), (0, 0)]
    pad_list.extend([(padding[i], padding[i]) for i in range(dimensionality)])
    padded_image = np.pad(img, tuple(pad_list),
                          mode="constant", constant_values=padWith)
    if alternate_input is not None:
        alternate_input = np.pad(alternate_input, tuple(pad_list),
                                 mode="constant", constant_values=padWith)

    ret = []
    for batch in range(N):
        tmp = []
        for in_channel in range(in_channels):
            ret2 = place_pools(padded_image[batch, in_channel],
                               kernel, stride, func,
                               alternate_input=alternate_input[batch, in_channel]
                               if alternate_input is not None else None)
            tmp.append(ret2)
        ret.append(tmp)
    ret = np.array(ret)
    return ret


def place_pools(img, kernel, stride, mode, offset=0, alternate_input=None):
    kernal = (np.array(img.shape) - np.array(kernel)) // np.array(stride) + 1
    kernels = np.zeros(tuple(kernal))

    for pos in generate_positions(img, stride, offset, min_space=np.array(kernel) - 1):
        slices = [slice(pos[i], pos[i] + kernel[i]) for i in range(len(img.shape))]
        if alternate_input is not None:
            alt = alternate_input[slices]
        else:
            alt = None
        kern = mode(img[slices], alternate_input=alt)
        kernels[tuple(np.array(pos) // stride)] = kern
    return kernels


def place_back_pools(img, kernel, stride, mode, dy, offset=0):
    ret = np.zeros(img.shape)

    for pos in generate_positions(img, stride, offset, min_space=np.array(kernel) - 1):
        slices = [slice(pos[i] if pos[i] > -1 else 0, pos[i] + kernel[i], 1)
                  for i in range(len(img.shape))]
        kern = mode(img[slices], dy[tuple(np.array(pos) // stride)])
        ret[slices] += kern[...]
    return ret


def max_pool(img, alternate_input=None):
    if alternate_input is None:
        return np.amax(img)
    else:
        return alternate_input.ravel()[np.argmax(img)]


def average_pool(img, alternate_input=None):
    if alternate_input is None:
        return np.average(img)
    else:
        return np.average(alternate_input)


def poolnim(original, dy, kernel, stride, padding, mode="max"):
    ret = np.zeros(original.shape)
    if mode is "max":
        func = back_max_pool
    elif mode is "average":
        func = back_average_pool

    _, _, in_dims = original.shape[0], original.shape[1], original.shape[2:]  # noqa
    dimensionality = len(in_dims)
    pad_list = [(0, 0), (0, 0)]
    pad_list.extend([(padding[i], padding[i]) for i in range(dimensionality)])
    padded_image = np.pad(original, tuple(pad_list),
                          mode="constant", constant_values=0)

    ret = np.zeros(original.shape)

    for batch in range(original.shape[0]):
        for in_channel in range(original.shape[1]):
            padding_slices = [slice(padding[i], padded_image.shape[2 + i] - padding[i])
                              for i in range(len(original.shape[2:]))]
            ret[batch, in_channel] = place_back_pools(
                padded_image[batch, in_channel], kernel, stride, func, dy[batch, in_channel])[padding_slices]
    return ret


def back_max_pool(img, dy):
    ret = np.zeros_like(img)
    ret.ravel()[np.argmax(img)] += dy
    return ret


def back_average_pool(img, dy):
    ret = np.zeros_like(img)
    ret += dy / ret.size
    return ret


def pad_dx(dx, original):
    ret = np.zeros_like(original)
    for p, v in np.ndenumerate(dx):
        ret[p] = v
    return ret


def generate_positions(img, stride=1, offset=0, min_space=0):
    pos = []
    if not isinstance(min_space, np.ndarray):
        min_space_list = []
        for _ in range(len(img.shape)):
            min_space_list.append(min_space)
    else:
        min_space_list = min_space
    for _ in range(len(img.shape)):
        pos.append(0)
    pos = np.array(pos, dtype=int)
    for p in enum_positions(pos, 0, len(pos), img.shape, stride, offset, min_space_list):
        yield tuple(p)


def enum_positions(pos_list, index, length, dist, stride, offset, min_space=0):
    pos_list[index] = -offset
    for x in range(-offset, dist[0] + offset - min_space[0], stride[0]):
        if index < length - 1:
            for pos in enum_positions(pos_list, index + 1, length, dist[1:], stride[1:], offset, min_space[1:]):
                yield pos
        else:
            yield pos_list
        pos_list[index] += stride[0]


def col2im(col, size, stride, padding, dilation=(1, 1)):
    in_h, in_w = size
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation
    N, channel, k_h, k_w, out_h, out_w = col.shape
    img = np.zeros((N, channel, in_h + 2 * p_h + s_h - 1,
                    in_w + 2 * p_w + s_w - 1), dtype=precision)
    for i in range(k_h):
        idh = i * d_h
        iu = idh + s_h * out_h

        for j in range(k_w):
            jdw = j * d_w
            ju = jdw + s_w * out_w
            img[:, :, idh:iu:s_h, jdw:ju:s_w] += col[:, :, k_h - 1 - i, k_w - 1 - j, :, :]

    im_shape = img.shape
    return img[:, :, p_h:im_shape[2] - (p_h + s_h - 1),
               p_w:im_shape[3] - (p_w + s_w - 1)]


def tuplize(x):
    return x if isinstance(x, tuple) else (x, x)


def roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(np.floor(size * stride))
    end = int(np.ceil((size + 1) * stride))

    start = min(max((start + roi_offset), 0), max_size)
    end = min(max((end + roi_offset), 0), max_size)

    return slice(start, end), end - start


def roi_pooling_slice_decode(size, stride, out_size, roi_offset):
    start = int(np.floor(float(size - roi_offset) / stride))
    end = int(np.ceil(float(size - roi_offset + 1) / stride))

    start = min(max(start, 0), out_size)
    end = min(max(end, 0), out_size)
    return start, end


def region_cordinates(roi, spatial_scale):
    idx, xmin, ymin, xmax, ymax = to_value(roi)
    idx = int(idx)
    xmin = int(round(xmin * spatial_scale))
    ymin = int(round(ymin * spatial_scale))
    xmax = int(round(xmax * spatial_scale))
    ymax = int(round(ymax * spatial_scale))
    return idx, xmin, ymin, xmax, ymax
