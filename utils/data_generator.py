import os
import random
import numpy as np
import scipy
import math
import cv2
import jpeg4py as jpeg
from scipy.misc import imread, imresize
from sklearn.utils import class_weight
from scipy.ndimage import rotate
#from skimage.exposure import adjust_gamma
from io import BytesIO
from PIL import Image

invGamma = 1 / 0.8
table1 = np.array([((i / 255.0) ** invGamma) * 255
                   for i in np.arange(0, 256)]).astype("uint8")
invGamma = 1 / 1.2
table2 = np.array([((i / 255.0) ** invGamma) * 255
                   for i in np.arange(0, 256)]).astype("uint8")


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma

    if abs(gamma - 0.8) < 0.01:
        global table1
        return cv2.LUT(image, table1)
    else:
        global table2
        return cv2.LUT(image, table2)


def center_crop(x, crop_size):
    centerh, centerw = int(x.shape[0] // 2), int(x.shape[1] // 2)
    halfh, halfw = int(crop_size[0] // 2), int(crop_size[1] // 2)
    return x[centerh - halfh:centerh + halfh, centerw - halfw:centerw + halfw, :]


def random_crop(x, crop_size):
    #h_im, w_img = x.shape[0], x.shape[1]
    left = random.randrange(x.shape[0] - crop_size[0])
    top = random.randrange(x.shape[1] - crop_size[1])
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return x[left:right, top:bottom, :]


def update_generator(batch_size, crop_size, num_classes, data_format):
    if data_format == 'channels_first':
        return np.zeros([batch_size, 3, crop_size[0], crop_size[1]]), \
            np.zeros([batch_size, num_classes]),\
            np.zeros(batch_size),\
            np.zeros([batch_size, 64])
    else:
        return np.zeros([batch_size, crop_size[0], crop_size[1], 3]),\
            np.zeros([batch_size, num_classes]),\
            np.zeros(batch_size),\
            np.zeros([batch_size, 64])


def make_sparse(num_classes, val):
    res = np.zeros(num_classes, dtype=np.float32)
    res[val] = 1.0
    return res


def get_labeled_data(datadir, classes):
    datainfo = []
    for i, cl in enumerate(classes):
        datainfo.extend(
            list(map(lambda filename: (os.path.join(
                datadir, cl, filename), i), os.listdir(os.path.join(datadir, cl))))
        )
    return datainfo


def get_generators(datadir, crop_size=(512, 512), batch_size=8,
                   val_dir=None, data_format='channels_last',
                   multiple_data=False, use_external_data=True,
                   use_pseudolabeling=False, use_manip_param=True):
    classes = ['Motorola-Nexus-6',
               'Motorola-Droid-Maxx',
               'Samsung-Galaxy-S4',
               'iPhone-6',
               'iPhone-4s',
               'Motorola-X',
               'LG-Nexus-5x',
               'Samsung-Galaxy-Note3',
               'HTC-1-M7',
               'Sony-NEX-7']
    # print(classes)
    num_classes = len(classes)
    datainfo = get_labeled_data(datadir, classes)
    if use_external_data:
        external_datadir = 'data/flickr_images'
        datainfo.extend(get_labeled_data(external_datadir, classes))
    if use_pseudolabeling:
        pseudo_datadir = 'data/pseudo2'
        datainfo.extend(get_labeled_data(pseudo_datadir, classes))

    random.shuffle(datainfo)

    train_class_weights = class_weight.compute_class_weight('balanced', np.array(
        list(range(num_classes))), np.array([info[1] for info in datainfo]))
    print(train_class_weights)

    def generator(datainfo, is_train, class_weights):
        while 1:
            random.shuffle(datainfo)
            X, Y, W, M = update_generator(
                batch_size, crop_size, num_classes, data_format)
            for i, (filename, cl) in enumerate(datainfo):
                try:
                    try:
                        img = jpeg.JPEG(filename).decode()
                    except Exception as e:
                        img = imread(filename, mode='RGB')
                    manip = 0.0
                    if not filename.startswith('data/pseudo'):
                        success = random.random() < 0.5
                        if success:
                            if random.random() < 0.5:
                                manip = 1.0
                                img = Image.fromarray(img)
                                out = BytesIO()
                                img.save(out, format='jpeg',
                                         quality=random.choice([random.randrange(70, 96), random.choice([70, 90])]))
                                img = imread(out)
                                del out
                            if random.random() < 0.5:
                                manip = 1.0
                                alpha = random.choice([0.5, 0.8, 1.5, 2.0])
                                if img.shape[0] * alpha > crop_size[0] and img.shape[1] * alpha > crop_size[1]:
                                    img = imresize(
                                        img, alpha, interp='bicubic')

                        if crop_size:
                            # if random.random() < 0.5:
                            #    img = Image.fromarray(img)
                            #    img = np.array(random_diagonal_crop(img, crop_size))
                            #    if img.shape[0] != crop_size[0] or img.shape[1] != crop_size[1]:
                            #        img = imresize(img, crop_size, interp='bicubic')
                            # else:
                            img = random_crop(img, crop_size)

                        if success:
                            if random.random() < 0.5:
                                manip = 1.0
                                img = adjust_gamma(
                                    img, gamma=random.choice([0.8, 1.2]))
                    else:
                        if 'manip' in filename:
                            manip = 1.0
                    if random.random() < 0.5:
                        img = np.rot90(img, k=random.choice(
                            [-1, 1]), axes=(0, 1))
                    if random.random() < 0.5:
                        img = np.fliplr(img)
                    if random.random() < 0.5:
                        img = np.flipud(img)
                    img = img.astype('float')
                    img -= [123.68, 116.779, 103.939]
                    img /= 255.0
                    if data_format == 'channels_first':
                        img = img.transpose([2, 0, 1])
                    label = make_sparse(num_classes, cl)
                    X[i % batch_size] = img
                    Y[i % batch_size] = label
                    W[i % batch_size] = class_weights[cl]
                    M[i % batch_size] = np.array([manip for _ in range(64)])
                    if (i + 1) % batch_size == 0:
                        if multiple_data:
                            yield [X, X, X, X], Y, W
                        elif use_manip_param:
                            yield {'input_1': X, 'aux_input': M}, Y, W
                        else:
                            yield X, Y, W
                        X, Y, W, M = update_generator(
                            batch_size, crop_size, num_classes, data_format)
                except Exception as e:
                    print(e)
    random.shuffle(datainfo)
    if val_dir:
        train_datainfo = datainfo
        val_datainfo = get_labeled_data(val_dir, classes)
        random.shuffle(val_datainfo)
        val_class_weights = class_weight.compute_class_weight('balanced', np.array(
            list(range(num_classes))), np.array([info[1] for info in val_datainfo]))
    else:
        thresh = int(200 / batch_size)
        train_datainfo = datainfo[thresh:]
        val_datainfo = datainfo[:thresh]
    return (generator(train_datainfo, True, train_class_weights), len(train_datainfo)), (generator(val_datainfo, False, val_class_weights), len(val_datainfo))


def test_generator(datadir, batch_size=8):
    datainfo = []
    classes = os.listdir(datadir)
    print(classes)
    num_classes = len(classes)
    for i, cl in enumerate(classes):
        datainfo.extend(
            list(map(lambda filename: (os.path.join(
                datadir, cl, filename), i), os.listdir(os.path.join(datadir, cl))))
        )
    while 1:
        random.shuffle(datainfo)
        X, Y = update_generator(batch_size, crop_size, num_classes)
        for i, (filename, cl) in enumerate(datainfo):
            img = imread(filename).astype('float')
            img -= [123.68, 116.779, 103.939]
            img /= 255.0
            if crop_size:
                img = center_crop(img, crop_size)
                assert(img.shape[0] == crop_size[0]
                       and img.shape[1] == crop_size[1])
            label = make_sparse(num_classes, cl)
            X[i % batch_size] = img
            Y[i % batch_size] = label
            if (i + 1) % batch_size == 0:
                yield X, Y
                X, Y = update_generator(batch_size, crop_size, num_classes)
