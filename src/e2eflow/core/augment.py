import numpy as np
import tensorflow as tf

from .spatial_transformer import transformer


def random_affine(tensors, *,
                  max_translation_x=0.0, max_translation_y=0.0,
                  max_rotation=0.0, min_scale=1.0, max_scale=1.0,
                  horizontal_flipping=False):
    """Applies geometric augmentations to a list of tensors.

    Each element in the list is augmented in the same way.
    For all elements, num_batch must be equal while height, width and channels
    may differ.
    """
    def _deg2rad(deg):
        return (deg * np.pi) / 180.0

    with tf.variable_scope('random_affine'):
        num_batch = tf.shape(tensors[0])[0]

        zero = tf.zeros([num_batch])
        one = tf.ones([num_batch])

        tx = tf.random_uniform([num_batch], -max_translation_x, max_translation_x)
        ty = tf.random_uniform([num_batch], -max_translation_y, max_translation_y)
        rot = tf.random_uniform([num_batch], -max_rotation, max_rotation)
        rad = _deg2rad(rot)
        scale = tf.random_uniform([num_batch], min_scale, max_scale)

        t1 = [[tf.cos(rad), -tf.sin(rad), tx],
              [tf.sin(rad), tf.cos(rad), ty]]
        t1 = tf.transpose(t1, [2, 0, 1])

        scale_x = scale
        if horizontal_flipping:
            flip = tf.random_uniform([num_batch], 0, 1)
            flip = tf.where(tf.greater(flip, 0.5), -one, one)
            scale_x = scale_x * flip

        t2 = [[scale_x, zero, zero],
              [zero, scale, zero],
              [zero, zero, one]]
        t2 = tf.transpose(t2, [2, 0, 1])

        t = tf.matmul(t1, t2)

        out = []
        for tensor in tensors:
            shape = tf.shape(tensor)
            tensor = transformer(tensor, t, (shape[1], shape[2]))
            out.append(tf.stop_gradient(tensor))
    return out


def random_photometric(ims, *,
                       noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                       brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                       min_gamma=1.0, max_gamma=1.0):
    """Applies photometric augmentations to a list of image batches.

    Each image in the list is augmented in the same way.
    For all elements, num_batch must be equal while height and width may differ.

    Args:
        ims: list of 3-channel image batches normalized to [0, 1].
        channel_mean: tensor of shape [3] which was used to normalize the pixel
            values ranging from 0 ... 255.

    Returns:
        Batch of normalized images with photometric augmentations. Has the same
        shape as the input batch.
    """

    with tf.variable_scope('random_photometric'):
        num_batch = tf.shape(ims[0])[0]

        contrast = tf.random_uniform([num_batch, 1], min_contrast, max_contrast)
        gamma = tf.random_uniform([num_batch, 1], min_gamma, max_gamma)
        gamma_inv = 1.0 / gamma
        colour = tf.random_uniform([num_batch, 3], min_colour, max_colour)
        if noise_stddev > 0.0:
            noise = tf.random_normal([num_batch, 1], stddev=noise_stddev)
        else:
            noise = tf.zeros([num_batch, 1])
        if brightness_stddev > 0.0:
            brightness = tf.random_normal([num_batch, 1],
                                          stddev=brightness_stddev)
        else:
            brightness = tf.zeros([num_batch, 1])

        out = []
        for im in ims:
            # Transpose to [height, width, num_batch, channels]
            im_re = tf.transpose(im, [1, 2, 0, 3])
            im_re = im_re
            im_re = (im_re * (contrast + 1.0) + brightness) * colour
            im_re = tf.maximum(0.0, tf.minimum(1.0, im_re))
            im_re = tf.pow(im_re, gamma_inv)

            im_re = im_re + noise

            # Subtract the mean again after clamping
            im_re = im_re

            im = tf.transpose(im_re, [2, 0, 1, 3])
            im = tf.stop_gradient(im)
            out.append(im)
        return out


def random_crop(tensors, size, seed=None, name=None):
    """Randomly crops multiple tensors (of the same shape) to a given size.

    Each tensor is cropped in the same way."""
    with tf.name_scope(name, "random_crop", [size]) as name:
        size = tf.convert_to_tensor(size, dtype=tf.int32, name="size")
        if len(tensors) == 2:
            shape = tf.minimum(tf.shape(tensors[0]), tf.shape(tensors[1]))
        else:
            shape = tf.shape(tensors[0])

        limit = shape - size + 1
        offset = tf.random_uniform(
           tf.shape(shape),
           dtype=size.dtype,
           maxval=size.dtype.max,
           seed=seed) % limit
        results = []
        for tensor in tensors:
            result = tf.slice(tensor, offset, size)
            results.append(result)
        return results
