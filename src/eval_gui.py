import os
import sys
import shutil

import tensorflow as tf
import numpy as np
import png

from e2eflow.core.flow_util import flow_to_color, flow_error_avg, outlier_pct
from e2eflow.core.flow_util import flow_error_image
from e2eflow.util import config_dict
from e2eflow.core.image_warp import image_warp
from e2eflow.kitti.input import KITTIInput
from e2eflow.kitti.data import KITTIData
from e2eflow.chairs.data import ChairsData
from e2eflow.chairs.input import ChairsInput
from e2eflow.sintel.data import SintelData
from e2eflow.sintel.input import SintelInput
from e2eflow.middlebury.input import MiddleburyInput
from e2eflow.middlebury.data import MiddleburyData
from e2eflow.core.unsupervised import unsupervised_loss
from e2eflow.core.input import resize_input, resize_output_crop, resize_output, resize_output_flow
from e2eflow.core.train import restore_networks
from e2eflow.ops import forward_warp, downsample
from e2eflow.gui import display
from e2eflow.core.losses import DISOCC_THRESH, occlusion, create_outgoing_mask
from e2eflow.util import convert_input_strings


tf.app.flags.DEFINE_string('dataset', 'kitti',
                            'Name of dataset to evaluate on. One of {kitti, sintel, chairs, mdb}.')
tf.app.flags.DEFINE_string('variant', 'train_2012',
                           'Name of variant to evaluate on.'
                           'If dataset = kitti, one of {train_2012, train_2015, test_2012, test_2015}.'
                           'If dataset = sintel, one of {train_clean, train_final}.'
                           'If dataset = mdb, one of {train, test}.')
tf.app.flags.DEFINE_string('ex', '',
                           'Experiment name(s) (can be comma separated list).')
tf.app.flags.DEFINE_integer('num', 10,
                            'Number of examples to evaluate. Set to -1 to evaluate all.')
tf.app.flags.DEFINE_integer('num_vis', 100,
                            'Number of evalutations to visualize. Set to -1 to visualize all.')
tf.app.flags.DEFINE_string('gpu', '0',
                           'GPU device to evaluate on.')
tf.app.flags.DEFINE_boolean('output_benchmark', False,
                            'Output raw flow files.')
tf.app.flags.DEFINE_boolean('output_visual', False,
                            'Output flow visualization files.')
tf.app.flags.DEFINE_boolean('output_backward', False,
                            'Output backward flow files.')
tf.app.flags.DEFINE_boolean('output_png', True, # TODO finish .flo output
                            'Raw output format to use with output_benchmark.'
                            'Outputs .png flow files if true, output .flo otherwise.')
FLAGS = tf.app.flags.FLAGS


NUM_EXAMPLES_PER_PAGE = 4


def write_rgb_png(z, path, bitdepth=8):
    z = z[0, :, :, :]
    with open(path, 'wb') as f:
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=bitdepth)
        z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
        writer.write(f, z2list)


def flow_to_int16(flow):
    _, h, w, _ = tf.unstack(tf.shape(flow))
    u, v = tf.unstack(flow, num=2, axis=3)
    r = tf.cast(tf.maximum(0.0, tf.minimum(u * 64.0 + 32768.0, 65535.0)), tf.uint16)
    g = tf.cast(tf.maximum(0.0, tf.minimum(v * 64.0 + 32768.0, 65535.0)), tf.uint16)
    b = tf.ones([1, h, w], tf.uint16)
    return tf.stack([r, g, b], axis=3)


def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    flow = flow[0, :, :, :]
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()


def _evaluate_experiment(name, input_fn, data_input):
    normalize_fn = data_input._normalize_image
    resized_h = data_input.dims[0]
    resized_w = data_input.dims[1]
    
    current_config = config_dict('../config.ini')
    exp_dir = os.path.join(current_config['dirs']['log'], 'ex', name)
    config_path = os.path.join(exp_dir, 'config.ini')
    if not os.path.isfile(config_path):
        config_path = '../config.ini'
    if not os.path.isdir(exp_dir) or not tf.train.get_checkpoint_state(exp_dir):
        exp_dir = os.path.join(current_config['dirs']['checkpoints'], name)
    config = config_dict(config_path)
    params = config['train']
    convert_input_strings(params, config_dict('../config.ini')['dirs'])
    dataset_params_name = 'train_' + FLAGS.dataset
    if dataset_params_name in config:
        params.update(config[dataset_params_name])
    ckpt = tf.train.get_checkpoint_state(exp_dir)
    if not ckpt:
        raise RuntimeError("Error: experiment must contain a checkpoint")
    ckpt_path = exp_dir + "/" + os.path.basename(ckpt.model_checkpoint_path)

    with tf.Graph().as_default(): #, tf.device('gpu:' + FLAGS.gpu):
        inputs = input_fn()
        im1, im2, input_shape = inputs[:3]
        truth = inputs[3:]

        height, width, _ = tf.unstack(tf.squeeze(input_shape), num=3, axis=0)
        im1 = resize_input(im1, height, width, resized_h, resized_w)
        im2 = resize_input(im2, height, width, resized_h, resized_w) # TODO adapt train.py

        _, flow, flow_bw = unsupervised_loss(
            (im1, im2),
            normalization=data_input.get_normalization(),
            params=params, augment=False, return_flow=True)

        im1 = resize_output(im1, height, width, 3)
        im2 = resize_output(im2, height, width, 3)
        flow = resize_output_flow(flow, height, width, 2)
        flow_bw = resize_output_flow(flow_bw, height, width, 2)

        flow_fw_int16 = flow_to_int16(flow)
        flow_bw_int16 = flow_to_int16(flow_bw)

        im1_pred = image_warp(im2, flow)
        im1_diff = tf.abs(im1 - im1_pred)
        #im2_diff = tf.abs(im1 - im2)

        #flow_bw_warped = image_warp(flow_bw, flow)

        if len(truth) == 4:
            flow_occ, mask_occ, flow_noc, mask_noc = truth
            flow_occ = resize_output_crop(flow_occ, height, width, 2)
            flow_noc = resize_output_crop(flow_noc, height, width, 2)
            mask_occ = resize_output_crop(mask_occ, height, width, 1)
            mask_noc = resize_output_crop(mask_noc, height, width, 1)

            #div = divergence(flow_occ)
            #div_bw = divergence(flow_bw)
            occ_pred = 1 - (1 - occlusion(flow, flow_bw)[0])
            def_pred = 1 - (1 - occlusion(flow, flow_bw)[1])
            disocc_pred = forward_warp(flow_bw) < DISOCC_THRESH
            disocc_fw_pred = forward_warp(flow) < DISOCC_THRESH
            image_slots = [((im1 * 0.5 + im2 * 0.5) / 255, 'overlay'),
                           (im1_diff / 255, 'brightness error'),
                           #(im1 / 255, 'first image', 1, 0),
                           #(im2 / 255, 'second image', 1, 0),
                           #(im2_diff / 255, '|first - second|', 1, 2),
                           (flow_to_color(flow), 'flow'),
                           #(flow_to_color(flow_bw), 'flow bw prediction'),
                           #(tf.image.rgb_to_grayscale(im1_diff) > 20, 'diff'),
                           #(occ_pred, 'occ'),
                           #(def_pred, 'disocc'),
                           #(disocc_pred, 'reverse disocc'),
                           #(disocc_fw_pred, 'forward disocc prediction'),
                           #(div, 'div'),
                           #(div < -2, 'neg div'),
                           #(div > 5, 'pos div'),
                           #(flow_to_color(flow_occ, mask_occ), 'flow truth'),
                           (flow_error_image(flow, flow_occ, mask_occ, mask_noc),
                            'flow error') #  (blue: correct, red: wrong, dark: occluded)
            ]

            # list of (scalar_op, title)
            scalar_slots = [(flow_error_avg(flow_noc, flow, mask_noc), 'EPE_noc'),
                            (flow_error_avg(flow_occ, flow, mask_occ), 'EPE_all'),
                            (outlier_pct(flow_noc, flow, mask_noc), 'outliers_noc'),
                            (outlier_pct(flow_occ, flow, mask_occ), 'outliers_all')]
        elif len(truth) == 2:
            flow_gt, mask = truth
            flow_gt = resize_output_crop(flow_gt, height, width, 2)
            mask = resize_output_crop(mask, height, width, 1)

            image_slots = [((im1 * 0.5 + im2 * 0.5) / 255, 'overlay'),
                           (im1_diff / 255, 'brightness error'),
                           (flow_to_color(flow), 'flow'),
                           (flow_to_color(flow_gt, mask), 'gt'),
            ]

            # list of (scalar_op, title)
            scalar_slots = [(flow_error_avg(flow_gt, flow, mask), 'EPE_all')]
        else:
            image_slots = [(im1 / 255, 'first image'),
                           #(im1_pred / 255, 'warped second image', 0, 1),
                           (im1_diff / 255, 'warp error'),
                           #(im2 / 255, 'second image', 1, 0),
                           #(im2_diff / 255, '|first - second|', 1, 2),
                           (flow_to_color(flow), 'flow prediction')]
            scalar_slots = []

        num_ims = len(image_slots)
        image_ops = [t[0] for t in image_slots]
        scalar_ops = [t[0] for t in scalar_slots]
        image_names = [t[1] for t in image_slots]
        scalar_names = [t[1] for t in scalar_slots]
        all_ops = image_ops + scalar_ops

        image_lists = []
        averages = np.zeros(len(scalar_ops))
        sess_config = tf.ConfigProto(allow_soft_placement=True)

        exp_out_dir = os.path.join('../out', name)
        if FLAGS.output_visual or FLAGS.output_benchmark:
            if os.path.isdir(exp_out_dir):
                shutil.rmtree(exp_out_dir)
            os.makedirs(exp_out_dir)
            shutil.copyfile(config_path, os.path.join(exp_out_dir, 'config.ini'))

        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            restore_networks(sess, params, ckpt, ckpt_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,
                                                   coord=coord)

            # TODO adjust for batch_size > 1 (also need to change image_lists appending)
            max_iter = FLAGS.num if FLAGS.num > 0 else None

            try:
                num_iters = 0
                while not coord.should_stop() and (max_iter is None or num_iters != max_iter):
                    all_results = sess.run([flow, flow_bw, flow_fw_int16, flow_bw_int16] + all_ops)
                    flow_fw_res, flow_bw_res, flow_fw_int16_res, flow_bw_int16_res = all_results[:4]
                    all_results = all_results[4:]
                    image_results = all_results[:num_ims]
                    scalar_results = all_results[num_ims:]
                    iterstr = str(num_iters).zfill(6)
                    if FLAGS.output_visual:
                        path_col = os.path.join(exp_out_dir, iterstr + '_flow.png')
                        path_overlay = os.path.join(exp_out_dir, iterstr + '_img.png')
                        path_error = os.path.join(exp_out_dir, iterstr + '_err.png')
                        write_rgb_png(image_results[0] * 255, path_overlay)
                        write_rgb_png(image_results[1] * 255, path_col)
                        write_rgb_png(image_results[2] * 255, path_error)
                    if FLAGS.output_benchmark:
                        path_fw = os.path.join(exp_out_dir, iterstr)
                        if FLAGS.output_png:
                            write_rgb_png(flow_fw_int16_res, path_fw  + '_10.png', bitdepth=16)
                        else:
                            write_flo(flow_fw_res, path_fw + '_10.flo')
                        if FLAGS.output_backward:
                            path_fw = os.path.join(exp_out_dir, iterstr + '_01.png')
                            write_rgb_png(flow_bw_int16_res, path_bw, bitdepth=16)
                    if num_iters < FLAGS.num_vis:
                        image_lists.append(image_results)
                    averages += scalar_results
                    if num_iters > 0:
                        sys.stdout.write('\r')
                    num_iters += 1
                    sys.stdout.write("-- evaluating '{}': {}/{}"
                                     .format(name, num_iters, max_iter))
                    sys.stdout.flush()
                    print()
            except tf.errors.OutOfRangeError:
                pass

            averages /= num_iters

            coord.request_stop()
            coord.join(threads)

    for t, avg in zip(scalar_slots, averages):
        _, scalar_name = t
        print("({}) {} = {}".format(name, scalar_name, avg))

    return image_lists, image_names


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    print("-- evaluating: on {} pairs from {}/{}"
          .format(FLAGS.num, FLAGS.dataset, FLAGS.variant))

    default_config = config_dict()
    dirs = default_config['dirs']

    if FLAGS.dataset == 'kitti':
        data = KITTIData(dirs['data'], development=True)
        data_input = KITTIInput(data, batch_size=1, normalize=False,
                                 dims=(384,1280))
        inputs = getattr(data_input, 'input_' + FLAGS.variant)()
    elif FLAGS.dataset == 'chairs':
        data = ChairsData(dirs['data'], development=True)
        data_input = ChairsInput(data, batch_size=1, normalize=False,
                                 dims=(384,512))
        if FLAGS.variant == 'test_2015' and FLAGS.num == -1:
            FLAGS.num = 200
        elif FLAGS.variant == 'test_2012' and FLAGS.num == -1:
            FLAGS.num = 195
    elif FLAGS.dataset == 'sintel':
        data = SintelData(dirs['data'], development=True)
        data_input = SintelInput(data, batch_size=1, normalize=False,
                                 dims=(512,1024))
    if FLAGS.variant in ['test_clean', 'test_final'] and FLAGS.num == -1:
        FLAGS.num = 552
    elif FLAGS.dataset == 'mdb':
        data = MiddleburyData(dirs['data'], development=True)
        data_input = MiddleburyInput(data, batch_size=1, normalize=False,
                                     dims=(512,640))
        if FLAGS.variant == 'test' and FLAGS.num == -1:
            FLAGS.num = 12

    input_fn = getattr(data_input, 'input_' + FLAGS.variant)

    results = []
    for name in FLAGS.ex.split(','):
        result, image_names = _evaluate_experiment(name, input_fn, data_input)
        results.append(result)

    display(results, image_names)





if __name__ == '__main__':
    tf.app.run()
