import os
import random

import numpy as np
import tensorflow as tf

from .augment import random_crop


def resize_input(t, height, width, resized_h, resized_w):
    # Undo old resizing and apply bilinear
    t = tf.reshape(t, [resized_h, resized_w, 3])
    t = tf.expand_dims(tf.image.resize_image_with_crop_or_pad(t, height, width), 0)
    return tf.image.resize_bilinear(t, [resized_h, resized_w])


def resize_output_crop(t, height, width, channels):
    _, oldh, oldw, c = tf.unstack(tf.shape(t))
    t = tf.reshape(t, [oldh, oldw, c])
    t = tf.image.resize_image_with_crop_or_pad(t, height, width)
    return tf.reshape(t, [1, height, width, channels])


def resize_output(t, height, width, channels):
    return tf.image.resize_bilinear(t, [height, width])


def resize_output_flow(t, height, width, channels):
    batch, old_height, old_width, _ = tf.unstack(tf.shape(t), num=4)
    t = tf.image.resize_bilinear(t, [height, width])
    u, v = tf.unstack(t, axis=3)
    u *= tf.cast(width, tf.float32) / tf.cast(old_width, tf.float32)
    v *= tf.cast(height, tf.float32) / tf.cast(old_height, tf.float32)
    return tf.reshape(tf.stack([u, v], axis=3), [batch, height, width, 2])


def frame_name_to_num(name):
    stripped = name.split('.')[0].lstrip('0')
    if stripped == '':
        return 0
    return int(stripped)


class Input():
    mean = [104.920005, 110.1753, 114.785955]
    stddev = 1 / 0.0039216

    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        assert len(dims) == 2
        self.data = data
        self.dims = dims
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.normalize = normalize
        self.skipped_frames = skipped_frames

    def _resize_crop_or_pad(self, tensor):
        height, width = self.dims
        # return tf.image.resize_bilinear(tf.expand_dims(tensor, 0), [height, width])
        return tf.image.resize_image_with_crop_or_pad(tensor, height, width)

    def _resize_image_fixed(self, image):
        height, width = self.dims
        return tf.reshape(self._resize_crop_or_pad(image), [height, width, 3])

    def _normalize_image(self, image):
        return (image - self.mean) / self.stddev

    def _preprocess_image(self, image):
        image = self._resize_image_fixed(image)
        if self.normalize:
            image = self._normalize_image(image)
        return image

    def _input_images(self, image_dir, hold_out_inv=None):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        image_dir = os.path.join(self.data.current_dir, image_dir)

        filenames_1 = []
        filenames_2 = []
        image_files = os.listdir(image_dir)
        image_files.sort()

        assert len(image_files) % 2 == 0, 'expected pairs of images'

        for i in range(len(image_files) // 2):
            filenames_1.append(os.path.join(image_dir, image_files[i * 2]))
            filenames_2.append(os.path.join(image_dir, image_files[i * 2 + 1]))

        if hold_out_inv is not None:
            filenames = list(zip(filenames_1, filenames_2))
            random.seed(0)
            random.shuffle(filenames)
            filenames = filenames[:hold_out_inv]

            filenames_1, filenames_2 = zip(*filenames)
            filenames_1 = list(filenames_1)
            filenames_2 = list(filenames_2)

        input_1 = read_png_image(filenames_1, 1)
        input_2 = read_png_image(filenames_2, 1)
        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)
        return tf.shape(input_1), image_1, image_2

    def _input_test(self, image_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def get_normalization(self):
        return self.mean, self.stddev

    def input_raw(self, swap_images=True, sequence=True,
                  needs_crop=True, shift=0, seed=0,
                  center_crop=False, skip=0):
        """Constructs input of raw data.

        Args:
            sequence: Assumes that image file order in data_dirs corresponds to
                temporal order, if True. Otherwise, assumes uncorrelated pairs of
                images in lexicographical ordering.
            shift: number of examples to shift the input queue by.
                Useful to resume training.
            swap_images: for each pair (im1, im2), also include (im2, im1)
            seed: seed for filename shuffling.
        Returns:
            image_1: batch of first images
            image_2: batch of second images
        """
        if not isinstance(skip, list):
            skip = [skip]

        data_dirs = self.data.get_raw_dirs()
        height, width = self.dims
        #assert batch_size % 2 == 0

        filenames = []
        for dir_path in data_dirs:
            files = os.listdir(dir_path)
            files.sort()
            if sequence:
                steps = [1 + s for s in skip]
                stops = [len(files) - s for s in steps]
            else:
                steps = [2]
                stops = [len(files)]
                assert len(files) % 2 == 0
            for step, stop in zip(steps, stops):
                for i in range(0, stop, step):
                    if self.skipped_frames and sequence:
                        assert step == 1
                        num_first = frame_name_to_num(files[i])
                        num_second = frame_name_to_num(files[i+1])
                        if num_first + 1 != num_second:
                            continue
                    fn1 = os.path.join(dir_path, files[i])
                    fn2 = os.path.join(dir_path, files[i + 1])
                    filenames.append((fn1, fn2))

        random.seed(seed)
        random.shuffle(filenames)
        print("Training on {} frame pairs.".format(len(filenames)))

        filenames_extended = []
        for fn1, fn2 in filenames:
            filenames_extended.append((fn1, fn2))
            if swap_images:
                filenames_extended.append((fn2, fn1))

        shift = shift % len(filenames_extended)
        filenames_extended = list(np.roll(filenames_extended, shift))


        filenames_1, filenames_2 = zip(*filenames_extended)
        filenames_1 = list(filenames_1)
        filenames_2 = list(filenames_2)

        with tf.variable_scope('train_inputs'):
            image_1 = read_png_image(filenames_1)
            image_2 = read_png_image(filenames_2)

            if needs_crop:
                #if center_crop:
                #    image_1 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
                #    image_2 = tf.image.resize_image_with_crop_or_pad(image_1, height, width)
                #else:
                image_1, image_2 = random_crop([image_1, image_2], [height, width, 3])
            else:
                image_1 = tf.reshape(image_1, [height, width, 3])
                image_2 = tf.reshape(image_2, [height, width, 3])

            if self.normalize:
                image_1 = self._normalize_image(image_1)
                image_2 = self._normalize_image(image_2)

            return tf.train.batch(
                [image_1, image_2],
                batch_size=self.batch_size,
                num_threads=self.num_threads)


def read_png_image(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for images."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames))
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_uint8 = tf.image.decode_png(value, channels=3)
    image = tf.cast(image_uint8, tf.float32)
    return image
