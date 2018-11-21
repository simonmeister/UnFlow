import tensorflow as tf
from ..ops import downsample as downsample_ops


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

def downsample(tensor, num):
    _,height, width,_ = tensor.shape.as_list()
    if height%2==0 and width%2==0:
        return downsample_ops(tensor, num)
    else:
        return tf.image.resize_area(tensor,tf.constant([int(height/num),int(width/num)]))
	
