import re
import os

import tensorflow as tf

import numpy as np

from .util import summarized_placeholder
from .image_warp import image_warp
from .flow_util import flow_to_color, flow_error_avg, outlier_pct
from .unsupervised import unsupervised_loss


def evaluate(batch_fn, params, normalization,
             summaries_dir, ckpt_dir, debug=False):
    """Evaluates all model checkpoints."""
    with tf.Graph().as_default():

        batch = batch_fn()
        truths = batch[2:]
        _, flow, _ = unsupervised_loss(
            batch[0:2], params=params, normalization=normalization,
            augment=False, return_flow=True)

        if debug:
            tf.summary.image('eval/warped', image_warp(batch[1], flow),
                             collections=['eval_images'])
            tf.summary.image('eval/flow', flow_to_color(flow),
                             collections=['eval_images'])

        values_ = []
        averages_ = []
        truth_tuples = []
        if len(truths) == 4:
            truth_tuples.append(('occluded', truths[0], truths[1]))
            truth_tuples.append(('non-occluded', truths[2], truths[3]))
        else:
            truth_tuples.append(('flow', truths[0], truths[1]))

        for name, gt_flow, mask in truth_tuples:
            error_ = flow_error_avg(gt_flow, flow, mask)
            error_avg_ = summarized_placeholder('AEE/' + name, key='eval_avg')
            outliers_ = outlier_pct(gt_flow, flow, mask)
            outliers_avg = summarized_placeholder('outliers/' + name,
                                                  key='eval_avg')
            if debug:
                tf.summary.image('flow-gt/' + name,
                                 flow_to_color(gt_flow, mask),
                                 collections=['eval_images'])
            values_.extend([error_, outliers_])
            averages_.extend([error_avg_, outliers_avg])

        losses = tf.get_collection('losses')
        for l in losses:
            values_.append(l)
            tensor_name = re.sub('tower_[0-9]*/', '', l.op.name)
            loss_avg_ = summarized_placeholder(tensor_name, key='eval_avg')
            averages_.append(loss_avg_)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt is None:
            print('Warning: no checkpoints to evaluate')
            return

        for ckpt_path in ckpt.all_model_checkpoint_paths:
            # Correct path for ckpts from different machine
            ckpt_path = ckpt_dir + "/" + os.path.basename(ckpt_path)
            with tf.Session() as sess:
                summary_writer = tf.summary.FileWriter(summaries_dir)
                saver = tf.train.Saver(tf.global_variables())

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                saver.restore(sess, ckpt_path)
                global_step = ckpt_path.split('/')[-1].split('-')[-1]

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,
                                                       coord=coord)
                averages = np.zeros(len(averages_))
                num_iters = 0

                try:
                    while not coord.should_stop():
                        if debug:
                            image_summary_ = tf.summary.merge_all('eval_images')
                            results = sess.run(values_ + [image_summary_])
                            values = results[0:-1]
                            image_summary = results[-1]
                            summary_writer.add_summary(image_summary, global_step)
                        else:
                            values = sess.run(values_)
                        averages += values
                        num_iters += 1
                except tf.errors.OutOfRangeError:
                    pass

                averages /= num_iters
                feed = {k: v for (k, v) in zip(averages_, averages)}

                summary_ = tf.summary.merge_all('eval_avg')
                summary = sess.run(summary_, feed_dict=feed)
                summary_writer.add_summary(summary, global_step)

                print("-- eval: i = {}, avg = {}".format(global_step, averages))

                coord.request_stop()
                coord.join(threads)
                summary_writer.close()
