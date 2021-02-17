# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import numpy as np


def get_appropriate_directory(directory):
    u"""Use to check 'directory' is a directory and ends with '/'

    :param directory: a directory path
    :type directory: str
    :return str: a directory path which ends with '/'

    :Example:
    >>> from utilities import get_appropriate_directory
    >>> get_appropriate_directory('hogehoge')
    'hogehoge/'
    >>> get_appropriate_directory('NOT_EXISTING_DIRECTORY')
    'input existing directory'
    """

    # check 'direcotry' is a directory
    if not os.path.isdir(directory):
        print('input existing directory')
        sys.exit()

    # append '/' if 'directory' does not end with that
    if not re.match('.*/$', directory):
        directory = directory + '/'

    return directory


def convert_class_to_onehot(class_list):
    u"""Use to create a one-hot vectors list from class_list

    :param class_list: sorted list of all the class names
    :type class_list: list
    :return list: list of one-hot vectors the index
                  of which corresponds to the index of class_list

    :Example:
    >>> from utilities import convert_class_to_onehot
    >>> class_list = ['apple', 'pen', 'pinapple']
    >>> convert_class_to_onehot(class_list)
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    """
    onehot_vectors = []
    for i in range(len(class_list)):
        temp = [0] * len(class_list)
        temp[-(i + 1)] = 1
        onehot_vectors.append(temp)
    return onehot_vectors


def is_image(filepath):
    u"""Use to check if the extention of 'filepath' is that of image files

    :param filepath: a path which tells a certain image file
    :type filepath: str
    :return boolean:

    :Example:
    >>> from utilities import is_image
    >>> img_filepath, noimg_filepath = './hogehoge.jpg', './foobar.txt'
    >>> is_image(img_filepath)
    True
    >>> is_image(noimg_filepath)
    False
    """
    matcher = re.compile(
        r'.*\.(JPG|jpg|JPEG|jpeg|PNG|png|TIFF|tiff|GIF|gif|BMP|bmp)$')
    if matcher.match(filepath):
        result = True
    else:
        result = False
    return result


def get_class_from_onehot(onehot_vector, class_list):
    u"""Use to get the corresponding name from a one-hot vector

    :param onehot_vector: one-hot vector
    :param class_list: a sorted list of all the class names
    :type onehot_vector: list (of int)
    :type class_list: list (of str)
    :return str: name of the class

    :Example:
    >>> from utilities import get_class_from_onehot
    >>> onehot_vectors = [[0, 1, 0], [1, 0, 0]]
    >>> class_list = ['apple', 'pen', 'pinapple']
    >>> transformed = [get_class_from_onehot(onehot, class_list) for onehot in onehot_vectors]
    >>> transformed
    ['pen', 'pinapple']
    """
    if max(onehot_vector) == 1:
        index = onehot_vector.index(1)
        return class_list[-(index + 1)]
    else:
        print("Please input one-hot vector")


def convert_minmax_to_wh(bndbox):
    u"""Use to transform [x_min, y_min, x_max, y_max] to [X(x_center), Y(y_center), W(width), H(height)].

    :param bndbox: bounding box which is expressed as [x_min, y_min, x_max, y_max]
    :type bndbox: list (of int)
    :return trans_list (of double):

    :Example:
    >>> from utililies import convert_minmax_to_wh
    >>> bndbox = [10, 20, 100, 80]
    >>> transformed = convert_minmax_to_wh(bndbox)
    >>> transformed
    [55.0, 50.0, 90.0, 60.0]
    """
    trans_list = [0, 0, 0, 0]

    # Check value
    if (len(bndbox) != 4):
        print("Input valid bounding box")
        sys.exit()
    if (bndbox[0] > bndbox[2]) | (bndbox[1] > bndbox[3]):
        print("inappropriate order. input [x_min, y_min, x_max, y_max]")
        sys.exit()
    trans_list[0] = (bndbox[0] + bndbox[2]) / 2.0
    trans_list[1] = (bndbox[1] + bndbox[3]) / 2.0
    trans_list[2] = (bndbox[2] - bndbox[0])
    trans_list[3] = (bndbox[3] - bndbox[1])

    return trans_list


def convert_wh_to_minmax(bndbox):
    u"""Use to transform [X(x_center), Y(y_center), W(width), H(height)] to [x_min, y_min, x_max, y_max].

    :param bndbox: bounding box which is expressed as [X(x_center), Y(y_center), W(width), H(height)]
    :type bndbox: list (of double)
    :return list (of int):

    :Example:
    >>> bndbox = [30.5, 60.0, 50.0, 80.0]
    >>> transformed = convert_wh_to_minmax(bndbox)
    >>> transformed
    [5, 20, 55, 100]
    """
    if len(bndbox) != 4:
        print("Input valid bounding box")
        sys.exit()

    trans_list = [int(bndbox[0] - bndbox[2] / 2),
                  int(bndbox[1] - bndbox[3] / 2),
                  int(bndbox[0] + bndbox[2] / 2),
                  int(bndbox[1] + bndbox[3] / 2)]
    return trans_list


def convert_coco_to_minmax(bndbox):
    u"""Use to transform [xmin, ymin, W, H] to [xmin, ymin, xmax, ymax]

    :param bndbox: bounding box which is expressed as [x_min, y_min, W(width), H(height)]
    :type bndbox: list (of double)
    :return list (of int):

    :Example:
    >>> from utilities import convert_coco_to_minmax
    >>> bndbox = [10.0, 20.0, 40.0, 60.0]
    >>> transformed = convert_coco_to_minmax(bndbox)
    >>> transformed
    [10, 20, 30, 50]
    """
    if len(bndbox) != 4:
        print("Input valid bounding box")
        sys.exit()

    trans_list = [int(bndbox[0]),
                  int(bndbox[1]),
                  int(bndbox[0] + bndbox[2]),
                  int(bndbox[1] + bndbox[3])]
    return trans_list


def get_num_images(Y_list):
    u"""Use to get number of Y_list.

    :param Y_list: result from load_for_detection.load_for_detection
    :type Y_list: list
    :result int: the number of the images in Y_list

    :Example:
    >>> from utilities import get_num_images
    >>> Y_list = [{'data': [{'bndbox': [1, 111, 58, 169], 'name': 'tvmonitor'},
                            {'bndbox': [371, 160, 476, 404], 'name': 'person'},
                            {'bndbox': [227, 99, 365, 404], 'name': 'person'},
                            {'bndbox': [146, 207, 263, 366], 'name': 'chair'}],
                   'filepath': 'VOC2012/2007_001717.jpg'},
                  {'data': [{'bndbox': [153, 22, 355, 342], 'name': 'sheep'}],
                   'filepath': 'VOC2012/2007_001872.jpg'}]
    >>> get_num_images(Y_list)
    2

    """
    num = len(Y_list)
    return num


def get_max_num_objects(Y_list):
    u"""Use to get the maximum number of the objects in a image.

    :param Y_list: result from load_for_detection.load_for_detection
    :type Y_list: list:
    :return int: number of the maximum number of the objects in one image

    :Example:
    >>> from utilities import get_max_num_objects
    >>> Y_list = [{'data': [{'bndbox': [1, 111, 58, 169], 'name': 'tvmonitor'},
                        {'bndbox': [371, 160, 476, 404], 'name': 'person'},
                        {'bndbox': [227, 99, 365, 404], 'name': 'person'},
                        {'bndbox': [146, 207, 263, 366], 'name': 'chair'}],
                   'filepath': 'VOC2012/2007_001717.jpg'},
                  {'data': [{'bndbox': [153, 22, 355, 342], 'name': 'sheep'}],
                   'filepath': 'VOC2012/2007_001872.jpg'}]
    >>> get_max_num_objects(Y_list)
    4

    """
    max_num = 1
    for data in Y_list:
        if len(data) > max_num:
            max_num = len(data)
    return max_num


def get_class_list(Y_list):
    u"""Use to get a list of the names sorted by alphabet that appear in Y_list.
    This is a legacy code.

    :param Y_list: result from load_for detection.load_for_detection
    :type Y_list: list
    :return list: sorted list of the class names that appear in Y_list

    :Example:
    >>> from utilities import get_class_list
    >>> Y_list = [{'data': [{'bndbox': [1, 111, 58, 169], 'name': 'tvmonitor'},
                            {'bndbox': [371, 160, 476, 404], 'name': 'person'},
                            {'bndbox': [227, 99, 365, 404], 'name': 'person'},
                            {'bndbox': [146, 207, 263, 366], 'name': 'chair'}],
                   'filepath': 'VOC2012/2007_001717.jpg'},
                  {'data': [{'bndbox': [153, 22, 355, 342], 'name': 'sheep'}],
                   'filepath': 'VOC2012/2007_001872.jpg'}]
    >>> class_list = get_class_list(Y_list)
    >>> class_list
    ['chair', 'person', 'sheep', 'tvmonitor']
    """
    class_list = []
    for data in Y_list:
        for subdata in data:
            name = subdata['name']
            if not (name in class_list):
                class_list.append(name)
    class_list.sort()
    return class_list


def convert_name_to_onehot(Y_list):
    u"""Use to convert all the name in Y_list (result from load_for_detection) to one hot data.
    names are sorted by alphabet.
    This is a legacy code.

    :param Y_list: result from load_for_detection.load_for_detection
    :type Y_list: list
    :return dictionary: the keys are names and the values are the corresponding one-hot vectors

    :Example:
    >>> from utilities import convert_name_to_onehot
    >>> Y_list = [{'data': [{'bndbox': [1, 111, 58, 169], 'name': 'tvmonitor'},
                            {'bndbox': [371, 160, 476, 404], 'name': 'person'},
                            {'bndbox': [227, 99, 365, 404], 'name': 'person'},
                            {'bndbox': [146, 207, 263, 366], 'name': 'chair'}],
                   'filepath': 'VOC2012/2007_001717.jpg'},
                  {'data': [{'bndbox': [153, 22, 355, 342], 'name': 'sheep'}],
                   'filepath': 'VOC2012/2007_001872.jpg'}]
    >>> result = convert_name_to_onehot(Y_list)
    >>> result
    {'chair': [0, 0, 0, 1], 'person', [0, 0, 1, 0], 'sheep': [0, 1, 0, 0], 'tvmonitor': [1, 0, 0, 0]}

    """
    class_list = get_class_list(Y_list)
    onehot_list = []
    zeros = [0] * len(class_list)
    for i in range(1, len(class_list) + 1):
        onehot = zeros[::]
        onehot[-i] = 1
        onehot_list.append(onehot)
    onehot_vectors = dict(zip(class_list, onehot_list))
    return onehot_vectors


def make_ndarray(Y_list, class_length):
    """Use to make ndarray from the result of load_for_detection

    :param Y_list: result from load_for_detection.load_for_detection
    :param class_length: length of the class_list
    :type Y_list: list
    :type class_length: int
    :return tuple(np.ndarray, int):
       | element1: ndarray([[X Y W H 0 0 0 1 0 X Y W H 0 0 0 0 1 ...], ...]])
       | element2: int

    :Example:
    >>> import numpy as np
    >>> from utilities import make_ndarray, get_max_num_length
    >>> Y_list = [[{'bndbox': convert_minmax_to_wh([1, 111, 58, 169]), 'name': [1, 0, 0, 0]},
                   {'bndbox': trans_minmax_to_WH([371, 160, 476, 404]), 'name': [0, 0, 1, 0]},
                   {'bndbox': trans_minmax_to_WH([227, 99, 365, 404]), 'name': [0, 0, 1, 0]},
                   {'bndbox': trans_minmax_to_WH([146, 207, 263, 366]), 'name': [0, 0, 0, 1]}],
                  [{'bndbox': trans_minmax_to_WH([153, 22, 355, 342]), 'name': [0, 1, 0, 0}]]
    >>> class_length = get_max_num_length(Y_list)
    >>> result = make_ndarray(Y_list, class_length)
    >>> result
    array([[  29.5,  140. ,   57. ,   58. ,    1. ,    0. ,    0. ,    0. ,
             423.5,  282. ,  105. ,  244. ,    0. ,    0. ,    1. ,    0. ,
             296. ,  251.5,  138. ,  305. ,    0. ,    0. ,    1. ,    0. ,
             204.5,  286.5,  117. ,  159. ,    0. ,    0. ,    0. ,    1. ],
           [ 254. ,  182. ,  202. ,  320. ,    0. ,    1. ,    0. ,    0. ,
               0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,
               0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,
               0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ]])
    """
    max_num = get_max_num_objects(Y_list)
    dim_cell = 4 + class_length
    array = np.zeros((len(Y_list), dim_cell * max_num))
    for i, data in enumerate(Y_list, 0):
        for j, subdata in enumerate(data, 0):
            # put [X Y W H]
            array[i, j * dim_cell:j * dim_cell +
                  4] = np.array(subdata['bndbox'])
            # put one-hot value
            array[i, j * dim_cell + 4:(j + 1) *
                  dim_cell] = np.array([subdata['name']])

    return array, class_length


def generate_colors_from_name_list(class_list):
    u"""Use to generate a dictionary whose key and value are a name and color value

    :param class_list: sorted list of all the class names
    :type class_list: list
    :return dict: {name1: rgb1, name2: rgb2, ...}

    :Example:
    >>> from utilities import generate_colors_from_name_list
    >>> class_list = ['apple', 'pen', 'pinapple']
    >>> rgb_dict = generate_colors_from_name_list(class_list)
    >>> rgb_dict
    {'apple': (105, 130, 29), 'pen': (246, 195, 65), 'pinapple': (197, 195, 92)}
    (values are randomly chosen)
    """
    import random

    rgb_dict = {}
    for i in range(0, len(class_list)):
        rgb = (random.randrange(255), random.randrange(
            255), random.randrange(255))
        rgb_dict[class_list[i]] = rgb
    return rgb_dict


def imshow_with_bndboxes(image, names, bndboxes, save_name=None):
    u"""Use to show an image with the corresponding bounding boxes and object names

    :param image: numpy array of an image
    :param names: list of the class names
    :param bndboxes: list of the bounding boxes (= [ [xmin, ymin, xmax, ymax], ... ]
    :param save_name: filepath to which the created image is to be saved
    :type image: np.ndarray
    :type names: list
    :type bndboxes: list
    :type save_name: str or None
    :return None:

    :Example:
    >>> import numpy as np
    >>> from PIL import Image
    >>> from utilities import imshow_with_bndboxes
    >>> from load_for_detection import load_from_xml
    >>> xmlpath = '../../test/test_dataset_test_load_for_detection_data/2007_000027.xml'
    >>> data = load_from_xml(xmlpath)
    >>> image = np.array(Image.open(data['filepath']))
    >>> names = [dct['name'] for dct in data['data']]
    >>> bndboxes = [dct['bndbox'] for dct in data['data']]
    >>> save_name = '2007_000027_with_bndboxes.jpg'
    >>> imshow_with_bndboxes(image, names, bndboxes, save_name)

    """
    import matplotlib.pyplot as plt

    # check
    if len(names) != len(bndboxes):
        print("length of names and that of bndboxes should be equal")
        sys.exit()

    # pltの調整
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.axis('off')
    plt.tight_layout()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

    # 重複なしのnamesを作成
    set_names = set(names)
    no_dupl_names = sorted([name for name in set_names])

    # colors の作成
    colors = plt.cm.hsv(np.linspace(0, 1, len(no_dupl_names) + 1)).tolist()

    # convert to int for showing with matplotlib
    image = image.astype(np.uint8).squeeze()
    color_map = None

    # for grayscale images
    if len(image.shape) == 2:
        color_map = 'gray'
    # show
    plt.imshow(image, cmap=color_map)
    currentAxis = plt.gca()
    for i in range(len(names)):
        name = names[i]
        bndbox = bndboxes[i]
        coords = (bndbox[0], bndbox[1]), bndbox[2] - \
            bndbox[0] + 1, bndbox[3] - bndbox[1] + 1
        color = colors[no_dupl_names.index(name)]
        currentAxis.add_patch(plt.Rectangle(
            *coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(bndbox[0], bndbox[1], name, bbox={
                         'facecolor': color, 'alpha': 0.5})

    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def imshow_batch(images, shape=None, output_path='./output.png'):
    u"""Use to show multiple images

    :param images: list (list(image1, image2, ...)) or np.ndarray (np.ndarray[numofimage][height][width][colors])
    :param shape: 2-dimentional list which represents [the number of vertical images, the number of horizontal images]
    :type images: list or np.ndarray
    :type shape: list
    :return None:

    :Example:
    >>> import numpy as np
    >>> from PIL import Image
    >>> from utilities import imshow_batch
    >>> image_path = ['../../doc/img_autodoc/image0001.jpg',
                      '../../doc/img_autodoc/image0002.jpg',
                      ...
                      '../../doc/img_autodoc/image0016.jpg']
    >>> images = [np.array(Image.open(path)) for path in image_path]
    >>> shape = [4, 4]
    >>> output_path = 'concatenated.jpg'
    >>> imshow_batch(images, shape, output_path)

    """

    import math
    import matplotlib.pyplot as plt
    color_map = None

    # images = list(ndarray1, ndarray2, ...)の時
    if isinstance(images, list):
        length = len(images)
        images = np.array(images)

    # images = ndarray[n][height][width][colors]の時
    elif isinstance(images, np.ndarray) and (len(images.shape) >= 3):
        length = images.shape[0]

    # for grayscale images
    if len(images[0].shape) == 2:
        color_map = 'gray'

    # convert to int for showing with matplotlib
    images = images.astype(np.uint8)

    # shapeが与えられていない時正方形にする
    if shape is None:
        width = int(math.ceil(math.sqrt(length)))
        height = int(math.ceil(length / width))

    # shapeの値のチェック
    elif (len(shape) != 2) or (shape[0] * shape[1] != length):
        print("shape[0] * shape[1] must be the number of the images.")
        sys.exit()
    else:
        height = shape[0]
        width = shape[1]

    for h in range(height):
        for w in range(width):
            if (w == 0) and (h * width + w < length):
                temp_concatenated_image = images[h * width + w]
            elif h * width + w < length:
                temp_concatenated_image = np.concatenate(
                    (temp_concatenated_image, images[h * width + w]), axis=1)
            elif (w == 0) and (h * width + w >= length):
                temp_concatenated_image = np.zeros(
                    images[0].shape, dtype=images[0].dtype)
            else:
                temp_concatenated_image = np.concatenate((temp_concatenated_image, np.zeros(
                    images[0].shape, dtype=images[0].dtype)), axis=1)
        if h == 0:
            concatenated_image = temp_concatenated_image
        else:
            concatenated_image = np.concatenate(
                (concatenated_image, temp_concatenated_image), axis=0)

    # display
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(concatenated_image, cmap=color_map)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def read_bndbox_from_ndarray(ndarray, num_class):
    u"""Use to read bounding boxes from a given ndarray.

    :param ndarray: result from make_ndarray
    :param num_class: the number of all the classes
    :type ndarray: ndarray
    :type num_class: int
    :return tuple(list, list):
       | element1: bounding boxes ( = [boxes_im1, boxes_im2, ...] )
       | element2: onehot_vectors ( = [onehot_im1, onehot_im2, ...])

    :Example:
    >>> import numpy as np
    >>> from utilities import read_bndbox_from_ndarray
    >>> ndarray = np.array([[  29.5,  140. ,   57. ,   58. ,    1. ,    0. ,    0. ,    0. ,
                              423.5,  282. ,  105. ,  244. ,    0. ,    0. ,    1. ,    0. ,
                              296. ,  251.5,  138. ,  305. ,    0. ,    0. ,    1. ,    0. ,
                              204.5,  286.5,  117. ,  159. ,    0. ,    0. ,    0. ,    1. ],
                            [ 254. ,  182. ,  202. ,  320. ,    0. ,    1. ,    0. ,    0. ,
                                0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,
                                0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,
                                0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ,    0. ]])
    >>> num_class = 4
    >>> bounding_boxes, onehot_vectors = read_bndbox_from_ndarray(ndarray, num_class)
    >>> bounding_boxes
    [[[  29.5,  140. ,   57. ,   58. ],
      [ 423.5,  282. ,  105. ,  244. ],
      [ 296. ,  251.5,  138. ,  305. ],
      [ 204.5,  286.5,  117. ,  159. ]],
     [[ 254. ,  182. ,  202. ,  320. ]]]
     >>> onehot_vectors
     >>> [[[1, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]],
          [[0, 1, 0, 0]]]
    """

    dim_cell = num_class + 4
    bounding_boxes = []
    onehot_vectors = []
    for row in ndarray:
        i = 0
        temp_bndbox = []
        temp_onehot = []
        # Until the number of the bounding boxes in a certain image
        while (not np.array_equal(row[i * dim_cell:(i + 1) * dim_cell],
                                  np.zeros(dim_cell))) and (i * dim_cell < len(row)):
            temp_bndbox.append(convert_wh_to_minmax(
                row[i * dim_cell:i * dim_cell + 4]))
            temp_onehot.append(list(row[i * dim_cell + 4:(i + 1) * dim_cell]))
            i += 1
        bounding_boxes.append(temp_bndbox)
        onehot_vectors.append(temp_onehot)
    #print("length of bouding_boxes:", len(bounding_boxes))
    return bounding_boxes, onehot_vectors


def build_yolo_labels(y, total_w, total_h, cells, classes):
    u"""Use to transform a list of objects per image into a image*cells*cells*(5+classes) matrix.
    Each cell in image can only be labeled for 1 object.

    "5" represents: objectness (0 or 1) and X Y W H

    :param y: np.ndarray ([batch][width][height])
    :param total_w: length of rows of y
    :param total_h: length of columns of y
    :param cells: grid size
    :param classes: length of class_list
    :type y: np.ndarray
    :type total_w: int
    :type total_h: int
    :type cells: int
    :type classes: int
    :return np.ndarray: [batch][cells][cells][1 + 4 + classes]

    :Example:
    >>> Input: 2 objects in first image, 5 classes
    >>> y[0] = X Y W H 0 1 0 0 0 X Y W H 0 0 0 1 0
        |---1st object----||---2nd object---|
    >>> Output: 7 * 7 cells * (1 + 4 + 5) per image
    >>> truth[0,0,0] = 1 X Y W H 0 1 0 0 0
        (cell 0,0 has first object)
    >>> truth[0,0,1] = 0 0 0 0 0 0 0 0 0 0
        (cell 0,1 has no object)
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
