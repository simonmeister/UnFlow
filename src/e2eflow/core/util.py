import tensorflow as tf


def summarized_placeholder(name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.placeholder(tf.float32, name=name)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p


def resize_area(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_area(tensor, [h, w]))


def resize_bilinear(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_bilinear(tensor, [h, w]))
