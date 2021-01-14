"""
Process an image that we can pass to our networks.
"""
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


def process_image(image_path, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image_path, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.0).astype(np.float32)

    return x


def preprocess_multires(image_path, target_shape):
    """ Given an image, process it and return high resolution and
    downsampeled arrays list."""
    h, w = target_shape

    context = load_img(image_path, target_size=(h, w),
                       interpolation="bilinear")
    context = img_to_array(context) - 96
    context = np.expand_dims(context, axis=0)

    fovea = crop_center(img_to_array(load_img(image_path))) - 96
    fovea = np.expand_dims(fovea, axis=0)
    return [fovea, context]


def crop_center(img):
    cropx = 89
    cropy = 89
    y, x, z = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty: starty + cropy, startx: startx + cropx] - 96
