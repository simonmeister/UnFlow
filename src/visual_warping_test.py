"""Code for playing with image warps on real data for debugging purposes."""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from e2eflow.core.image_warp import image_warp
from e2eflow.kitti.input import _read_flow, _read_image

HEIGHT = 320
WIDTH = 1152

def _resize_crop_or_pad(tensor):
    return tf.image.resize_image_with_crop_or_pad(tensor, HEIGHT, WIDTH)


def _preprocess_image(image):
    return tf.reshape(_resize_crop_or_pad(image), [HEIGHT, WIDTH, 3])

def _preprocess_flow(gt):
    flow, mask = gt
    # Reshape to tell tensorflow we know the size statically
    flow = tf.reshape(_resize_crop_or_pad(flow), [HEIGHT, WIDTH, 2])
    mask = tf.reshape(_resize_crop_or_pad(mask), [HEIGHT, WIDTH, 1])
    return flow, mask

mean = [104.920005, 110.1753, 114.785955]
stddev = 1 / 0.0039216
base_dir = "../data/data_stereo_flow/training/"
im1_file = [base_dir + 'colored_0/000002_10.png', base_dir + 'colored_0/000000_10.png']
im2_file = [base_dir + 'colored_0/000002_11.png', base_dir + 'colored_0/000000_11.png']
flow_file = [base_dir + 'flow_occ/000002_10.png', base_dir + 'flow_occ/000000_10.png']
im1 = (_preprocess_image(_read_image(im1_file)) - mean) / stddev
im2 = (_preprocess_image(_read_image(im2_file)) - mean) / stddev

ARTIFICIAL_FLOW = False
if ARTIFICIAL_FLOW:
    mask = tf.ones([HEIGHT, WIDTH, 1])
    # apparently.. the warping brakes when we add a fractional part
    flow = tf.ones([HEIGHT, WIDTH, 2]) + 40.5
else:
    flow, mask = _preprocess_flow(_read_flow(flow_file))

im1_batch, im2_batch, flow_batch, mask_batch = tf.train.batch(
    [im1, im2, flow, mask],
    batch_size=2)
im1_pred = image_warp(im2_batch, flow_batch)

if ARTIFICIAL_FLOW:
    diff = im2_batch - im1_pred # show diff between original # 2 and warped # 2
else:
    diff = im1_batch - im1_pred # show diff between original # 1 and warped # 2

# make sure to compare only fields which have a valid flow vector
diff = tf.multiply(diff, mask_batch)
error = tf.reduce_sum(tf.sqrt(tf.square(diff))) / (3 * tf.reduce_sum(mask_batch))

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
merged = tf.summary.merge_all()
sess = tf.Session()
writer = tf.summary.FileWriter('../log/exp', sess.graph)
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

error_value, im1_value, im2_value, im1_pred_value, diff_value, mask_value, flow_value, summary = sess.run(
    [error, im1_batch, im2_batch, im1_pred, diff, mask_batch, flow_batch, merged],
    options=run_options,
    run_metadata=run_metadata)
writer.add_summary(summary)

print("-- warp error: {}".format(error_value))
print(np.mean(flow_value))
plt.figure()
plt.title("im1_value")
plt.imshow(im1_value[0, :, :, :] + np.asarray(mean) / stddev)
plt.figure()
plt.title("im2_value")
plt.imshow(im2_value[0, :, :, :] + np.asarray(mean) / stddev)
plt.figure()
plt.title("im1_pred_value")
plt.imshow(im1_pred_value[0, :, :, :] + np.asarray(mean) / stddev)
plt.figure()
plt.title("diff_value")
plt.imshow(diff_value[0, :, :, :] + np.asarray(mean) / stddev)
plt.figure()
plt.title("mask_value")
plt.imshow(mask_value[0, :, :, 0], cmap='gray')
plt.show()
