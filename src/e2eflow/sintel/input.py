import os
import sys

import numpy as np
import tensorflow as tf

from ..core.input import read_png_image, Input


def _read_flow(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth flow files."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    value = tf.reshape(value, [1])
    value_width = tf.substr(value, 4, 4)
    value_height = tf.substr(value, 8, 4)
    width = tf.reshape(tf.decode_raw(value_width, out_type=tf.int32), [])
    height = tf.reshape(tf.decode_raw(value_height, out_type=tf.int32), [])

    value_flow = tf.substr(value, 12, 8 * 436 * 1024)
    flow = tf.decode_raw(value_flow, out_type=tf.float32)

    return tf.reshape(flow, [436, 1024, 2])


def _read_binary(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth binary files."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    value_decoded = tf.image.decode_png(value, channels=1)
    return tf.cast(value_decoded, tf.float32)


def _get_filenames(parent_dir, ignore_last=False):
    filenames = []
    for sub_name in sorted(os.listdir(parent_dir)):
        sub_dir = os.path.join(parent_dir, sub_name)
        sub_filenames = os.listdir(sub_dir)
        sub_filenames.sort()
        if ignore_last:
            sub_filenames = sub_filenames[:-1]
        for filename in sub_filenames:
            filenames.append(os.path.join(sub_dir, filename))

    return filenames


class SintelInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize)

    def _preprocess_flow(self, t, channels):
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        return tf.reshape(self._resize_crop_or_pad(t), [height, width, channels])

    def _input_images(self, image_dir):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        image_dir = os.path.join(self.data.current_dir, image_dir)

        filenames_1 = []
        filenames_2 = []

        for sub_name in sorted(os.listdir(image_dir)):
            sub_dir = os.path.join(image_dir, sub_name)
            sub_filenames = os.listdir(sub_dir)
            sub_filenames.sort()
            for i in range(len(sub_filenames) - 1):
                filenames_1.append(os.path.join(sub_dir, sub_filenames[i]))
                filenames_2.append(os.path.join(sub_dir, sub_filenames[i + 1]))

        input_1 = read_png_image(filenames_1, 1)
        input_2 = read_png_image(filenames_2, 1)
        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)
        return tf.shape(input_1), image_1, image_2

    def _input_flow(self):
        flow_dir = os.path.join(self.data.current_dir, 'sintel/training/flow')
        invalid_dir = os.path.join(self.data.current_dir, 'sintel/training/invalid')
        occ_dir = os.path.join(self.data.current_dir, 'sintel/training/occlusions')
        flow_files = _get_filenames(flow_dir)
        invalid_files = _get_filenames(invalid_dir, ignore_last=True)
        occ_files = _get_filenames(occ_dir)

        assert len(flow_files) == len(invalid_files) == len(occ_files)

        flow = self._preprocess_flow(_read_flow(flow_files, 1), 2)
        invalid = self._preprocess_flow(_read_binary(invalid_files), 1)
        occ = self._preprocess_flow(_read_binary(occ_files), 1)

        flow_occ = flow
        flow_noc = flow * (1 - occ)
        mask_occ = (1 - invalid)
        mask_noc = mask_occ * (1 - occ)

        return flow_occ, mask_occ, flow_noc, mask_noc

    def _input_train(self, image_dir):
        input_shape, im1, im2 = self._input_images(image_dir)
        flow_occ, mask_occ, flow_noc, mask_noc = self._input_flow()
        return tf.train.batch(
            [im1, im2, input_shape, flow_occ, mask_occ, flow_noc, mask_noc],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_train_clean(self):
        return self._input_train('sintel/training/clean')

    def input_train_final(self):
        return self._input_train('sintel/training/final')

    def input_test_clean(self):
        input_shape, im1, im2 = self._input_images('sintel/test/clean')
        return tf.train.batch(
           [im1, im2, input_shape],
           batch_size=self.batch_size,
           num_threads=self.num_threads,
           allow_smaller_final_batch=True)

    def input_test_final(self):
        input_shape, im1, im2 = self._input_images('sintel/test/final')
        return tf.train.batch(
           [im1, im2, input_shape],
           batch_size=self.batch_size,
           num_threads=self.num_threads,
           allow_smaller_final_batch=True)
