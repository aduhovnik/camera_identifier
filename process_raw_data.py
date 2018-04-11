import cv2
from PIL import ImageTk, Image


DEFAULT_CROP_SIZE = 512


def get_crop(image_path, CROP_SIZE=DEFAULT_CROP_SIZE, save_to=None):
    img = cv2.imread(image_path)

    h, w, _ = img.shape

    if h < CROP_SIZE:
        h, w = CROP_SIZE, int(w * (CROP_SIZE * 1. / h))

    if w < CROP_SIZE:
        w, h = CROP_SIZE, int(h * (CROP_SIZE * 1. / w))

    res = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    y0, x0 = (res.shape[0] - CROP_SIZE) // 2, (res.shape[1] - CROP_SIZE) // 2

    crop_img = res[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]

    if save_to:
        cv2.imwrite(save_to, crop_img)

    return crop_img


def resize(image_path, MAX_SIZE=600):
    img = cv2.imread(image_path)

    h, w, _ = img.shape

    if h > w:
        h, w = MAX_SIZE, int(w * (MAX_SIZE * 1. / h))
    else:
        w, h = MAX_SIZE, int(h * (MAX_SIZE * 1. / w))

    res = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(image_path, res)

    return res