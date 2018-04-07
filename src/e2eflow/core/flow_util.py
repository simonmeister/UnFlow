import numpy as np
import tensorflow as tf


def atan2(y, x):
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.greater_equal(y,0.0)),
                      tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.less(y,0.0)),
                      tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)),
                      np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)),
                      -np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0),tf.equal(y,0.0)),
                      np.nan * tf.zeros_like(x), angle)
    return angle


def flow_to_color(flow, mask=None, max_flow=None):
    """Converts flow to 3-channel color image.

    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    """
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(max_flow, 1)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask


def flow_error_image(flow_1, flow_2, mask_occ, mask_noc=None, log_colors=True):
    """Visualize the error between two flows as 3-channel color image.

    Adapted from the KITTI C++ devkit.

    Args:
        flow_1: first flow of shape [num_batch, height, width, 2].
        flow_2: second flow (ground truth)
        mask_occ: flow validity mask of shape [num_batch, height, width, 1].
            Equals 1 at (occluded and non-occluded) valid pixels.
        mask_noc: Is 1 only at valid pixels which are not occluded.
    """
    mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
    diff_sq = (flow_1 - flow_2) ** 2
    diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keepdims=True))
    if log_colors:
        num_batch, height, width, _ = tf.unstack(tf.shape(flow_1))
        colormap = [
            [0,0.0625,49,54,149],
            [0.0625,0.125,69,117,180],
            [0.125,0.25,116,173,209],
            [0.25,0.5,171,217,233],
            [0.5,1,224,243,248],
            [1,2,254,224,144],
            [2,4,253,174,97],
            [4,8,244,109,67],
            [8,16,215,48,39],
            [16,1000000000.0,165,0,38]]
        colormap = np.asarray(colormap, dtype=np.float32)
        colormap[:, 2:5] = colormap[:, 2:5] / 255
        mag = tf.sqrt(tf.reduce_sum(tf.square(flow_2), 3, keepdims=True))
        error = tf.minimum(diff / 3, 20 * diff / mag)
        im = tf.zeros([num_batch, height, width, 3])
        for i in range(colormap.shape[0]):
            colors = colormap[i, :]
            cond = tf.logical_and(tf.greater_equal(error, colors[0]),
                                  tf.less(error, colors[1]))
            im = tf.where(tf.tile(cond, [1, 1, 1, 3]),
                           tf.ones([num_batch, height, width, 1]) * colors[2:5],
                           im)
        im = tf.where(tf.tile(tf.cast(mask_noc, tf.bool), [1, 1, 1, 3]),
                       im, im * 0.5)
        im = im * mask_occ
    else:
        error = (tf.minimum(diff, 5) / 5) * mask_occ
        im_r = error # errors in occluded areas will be red
        im_g = error * mask_noc
        im_b = error * mask_noc
        im = tf.concat(axis=3, values=[im_r, im_g, im_b])
    return im


def flow_error_avg(flow_1, flow_2, mask):
    """Evaluates the average endpoint error between flow batches."""
    with tf.variable_scope('flow_error_avg'):
        diff = euclidean(flow_1 - flow_2) * mask
        error = tf.reduce_sum(diff) / tf.reduce_sum(mask)
        return error


def outlier_ratio(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    diff = euclidean(gt_flow - flow) * mask
    if relative is not None:
        threshold = tf.maximum(threshold, euclidean(gt_flow) * relative)
        outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
    else:
        outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
    ratio = tf.reduce_sum(outliers) / tf.reduce_sum(mask)
    return ratio


def outlier_pct(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    frac = outlier_ratio(gt_flow, flow, mask, threshold, relative) * 100
    return frac


def euclidean(t):
    return tf.sqrt(tf.reduce_sum(t ** 2, [3], keepdims=True))
