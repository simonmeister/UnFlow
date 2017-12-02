import os
import sys

import numpy as np
import tensorflow as tf
import random

from ..core.input import read_png_image, Input
from ..core.augment import random_crop


def _read_flow(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    gt_uint16 = tf.image.decode_png(value, dtype=tf.uint16)
    gt = tf.cast(gt_uint16, tf.float32)
    flow = (gt[:, :, 0:2] - 2 ** 15) / 64.0
    mask = gt[:, :, 2:3]
    return flow, mask


class KITTIInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _preprocess_flow(self, gt):
        flow, mask = gt
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        flow = tf.reshape(self._resize_crop_or_pad(flow), [height, width, 2])
        mask = tf.reshape(self._resize_crop_or_pad(mask), [height, width, 1])
        return flow, mask

    def _input_flow(self, flow_dir, hold_out_inv):
        flow_dir_occ = os.path.join(self.data.current_dir, flow_dir, 'flow_occ')
        flow_dir_noc = os.path.join(self.data.current_dir, flow_dir, 'flow_noc')
        filenames_gt_occ = []
        filenames_gt_noc = []
        flow_files_occ = os.listdir(flow_dir_occ)
        flow_files_occ.sort()
        flow_files_noc = os.listdir(flow_dir_noc)
        flow_files_noc.sort()

        if hold_out_inv is not None:
            random.seed(0)
            random.shuffle(flow_files_noc)
            flow_files_noc = flow_files_noc[:hold_out_inv]

            random.seed(0)
            random.shuffle(flow_files_occ)
            flow_files_occ = flow_files_occ[:hold_out_inv]

        assert len(flow_files_noc) == len(flow_files_occ)

        for i in range(len(flow_files_occ)):
            filenames_gt_occ.append(os.path.join(flow_dir_occ,
                                                 flow_files_occ[i]))
            filenames_gt_noc.append(os.path.join(flow_dir_noc,
                                                 flow_files_noc[i]))

        flow_occ, mask_occ = self._preprocess_flow(
            _read_flow(filenames_gt_occ, 1))
        flow_noc, mask_noc = self._preprocess_flow(
            _read_flow(filenames_gt_noc, 1))
        return flow_occ, mask_occ, flow_noc, mask_noc

    def _input_train(self, image_dir, flow_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        flow_occ, mask_occ, flow_noc, mask_noc = self._input_flow(flow_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape, flow_occ, mask_occ, flow_noc, mask_noc],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_train_gt(self, hold_out):
        img_dirs = ['data_scene_flow/training/image_2',
                    'data_stereo_flow/training/colored_0']
        gt_dirs = ['data_scene_flow/training/flow_occ',
                   'data_stereo_flow/training/flow_occ']

        height, width = self.dims

        filenames = []
        for img_dir, gt_dir in zip(img_dirs, gt_dirs):
            dataset_filenames = []
            img_dir = os.path.join(self.data.current_dir, img_dir)
            gt_dir = os.path.join(self.data.current_dir, gt_dir)
            img_files = os.listdir(img_dir)
            gt_files = os.listdir(gt_dir)
            img_files.sort()
            gt_files.sort()
            assert len(img_files) % 2 == 0 and len(img_files) / 2 == len(gt_files)

            for i in range(len(gt_files)):
                fn_im1 = os.path.join(img_dir, img_files[2 * i])
                fn_im2 = os.path.join(img_dir, img_files[2 * i + 1])
                fn_gt = os.path.join(gt_dir, gt_files[i])
                dataset_filenames.append((fn_im1, fn_im2, fn_gt))

            random.seed(0)
            random.shuffle(dataset_filenames)
            dataset_filenames = dataset_filenames[hold_out:]
            filenames.extend(dataset_filenames)

        random.seed(0)
        random.shuffle(filenames)

        #shift = shift % len(filenames)
        #filenames_ = list(np.roll(filenames, shift))

        fns_im1, fns_im2, fns_gt = zip(*filenames)
        fns_im1 = list(fns_im1)
        fns_im2 = list(fns_im2)
        fns_gt = list(fns_gt)

        im1 = read_png_image(fns_im1)
        im2 = read_png_image(fns_im2)
        flow_gt, mask_gt = _read_flow(fns_gt)

        gt_queue = tf.train.string_input_producer(fns_gt,
            shuffle=False, capacity=len(fns_gt), num_epochs=None)
        reader = tf.WholeFileReader()
        _, gt_value = reader.read(gt_queue)
        gt_uint16 = tf.image.decode_png(gt_value, dtype=tf.uint16)
        gt = tf.cast(gt_uint16, tf.float32)

        im1, im2, gt = random_crop([im1, im2, gt],
                                   [height, width, 3])
        flow_gt = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask_gt = gt[:, :, 2:3]

        if self.normalize:
            im1 = self._normalize_image(im1)
            im2 = self._normalize_image(im2)

        return tf.train.batch(
            [im1, im2, flow_gt, mask_gt],
            batch_size=self.batch_size,
            num_threads=self.num_threads)

    def input_train_2015(self, hold_out_inv=None):
        return self._input_train('data_scene_flow/training/image_2',
                                 'data_scene_flow/training',
                                 hold_out_inv)

    def input_test_2015(self, hold_out_inv=None):
        return self._input_test('data_scene_flow/testing/image_2', hold_out_inv)

    def input_train_2012(self, hold_out_inv=None):
        return self._input_train('data_stereo_flow/training/colored_0',
                                 'data_stereo_flow/training',
                                 hold_out_inv)

    def input_test_2012(self, hold_out_inv=None):
        return self._input_test('data_stereo_flow/testing/colored_0', hold_out_inv)
