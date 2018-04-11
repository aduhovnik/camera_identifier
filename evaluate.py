import os
import csv
import tempfile
import time
import numpy as np
import keras
import json
import random
import keras.backend as K
K.set_learning_phase(0)
from keras.models import load_model, Model
from keras.layers import Dense
from keras.utils.generic_utils import CustomObjectScope
from scipy.misc import imread
from collections import defaultdict


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def eval(model_path, test_path, aux_input=True, TTA=True):
    custom_objects = {'relu6': keras.applications.mobilenet.relu6,
                      'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}
    model = load_model(model_path, custom_objects=custom_objects)
    if TTA:
        weights = model.layers[-1].get_weights()
        x = model.layers[-2].output
        x = Dense(10, name='logits')(x)
        model = Model(input=model.inputs, output=x)
        model.layers[-1].set_weights(weights)

        # tta_funcs = [[np.fliplr, []],  [np.flipud, []],
        #              [np.rot90, [1, (0, 1)]], [np.flipud, [-1, (0, 1)]]]
        # used_tta_func = []

        # def apply_tta_func(inp):
        #     for func, pargs in tta_funcs:
        #         if (func, pargs) not in used_tta_func:
        #             used_tta_func.append((func, pargs))
        #             inp_res = func(inp, *pargs)
        #             apply_tta_func(inp_res)
        #             yield inp_res
        #             used_tta_func.pop()

    filenames = os.listdir(test_path)
    results = defaultdict(dict)

    print('Total {} images'.format(len(filenames)))

    labels = ['Motorola-Nexus-6',
              'Motorola-Droid-Maxx',
              'Samsung-Galaxy-S4',
              'iPhone-6',
              'iPhone-4s',
              'Motorola-X',
              'LG-Nexus-5x',
              'Samsung-Galaxy-Note3',
              'HTC-1-M7',
              'Sony-NEX-7']

    with open('csv/res_{}.csv'.format(model_path), 'w') as csvfile:
        res_writer = csv.writer(csvfile)
        res_writer.writerow(['fname', 'camera'])
        all_logits = np.zeros([2640, 10])
        for i, filename in enumerate(filenames):
            imgs = []
            img = imread(os.path.join(test_path, filename)).astype('float')
            imgs.append(img)
            if TTA:
                imgs.extend([
                    np.fliplr(img),
                    np.flipud(img),
                    np.rot90(img, 1, (0, 1)),
                    np.rot90(img, -1, (0, 1))
                ])
            imgs = list(
                map(lambda x: (x - [123.68, 116.779, 103.939]) / 255.0, imgs))
            inp = np.array(imgs)
            if aux_input:
                manip = float('manip' in filename)
                manip = np.array([manip for _ in range(64)])[np.newaxis, :]
                manip = np.repeat(manip, len(imgs), axis=0)
                pred = model.predict({'input_1': inp, 'aux_input': manip})
            else:
                pred = model.predict(inp)
            logits = np.mean(pred, axis=0)
            all_logits[i] = logits
            pred = softmax(logits)
            label, score = np.argmax(pred), np.max(pred)
            results[filename]['label'] = str(label)
            results[filename]['score'] = str(score)
            res_writer.writerow([filename, labels[label]])
            print(i, ' ', filename, ' ', score)
    np.save('logits.npy', all_logits)
    print('Results: ', results)
    #with open('scores/scores-{}.json'.format(model_path.split('/')[1].split('.')[1]), 'w') as f:
    #    json.dump(results, f)


def get_model(model_path='mobilenet_weights.09-0.13.h5', TTA=True):
    custom_objects = {'relu6': keras.applications.mobilenet.relu6,
                      'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}
    model = load_model(model_path, custom_objects=custom_objects)
    if TTA:
        weights = model.layers[-1].get_weights()
        x = model.layers[-2].output
        x = Dense(10, name='logits')(x)
        model = Model(input=model.inputs, output=x)
        model.layers[-1].set_weights(weights)
    return model


def process_one_file(filepath, model, TTA=True, aux_input=True):
    labels = ['Motorola-Nexus-6',
              'Motorola-Droid-Maxx',
              'Samsung-Galaxy-S4',
              'iPhone-6',
              'iPhone-4s',
              'Motorola-X',
              'LG-Nexus-5x',
              'Samsung-Galaxy-Note3',
              'HTC-1-M7',
              'Sony-NEX-7']

    imgs = []
    img = imread(filepath).astype('float')
    imgs.append(img)
    if TTA:
        imgs.extend([
            np.fliplr(img),
            np.flipud(img),
            np.rot90(img, 1, (0, 1)),
            np.rot90(img, -1, (0, 1))
        ])
    imgs = list(
        map(lambda x: (x - [123.68, 116.779, 103.939]) / 255.0, imgs))
    inp = np.array(imgs)
    if aux_input:
        manip = float('manip' in filepath)
        manip = np.array([manip for _ in range(64)])[np.newaxis, :]
        manip = np.repeat(manip, len(imgs), axis=0)
        pred = model.predict({'input_1': inp, 'aux_input': manip})
    else:
        pred = model.predict(inp)
    logits = np.mean(pred, axis=0)
    pred = softmax(logits)
    label, score = np.argmax(pred), np.max(pred)
    return labels[label], {labels[i]: val for i, val in enumerate(pred)}



if __name__ == '__main__':
    eval('mobilenet_weights.09-0.13.h5', 'data/test1')
