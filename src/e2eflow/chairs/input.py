import os

import numpy as np
import tensorflow as tf

from ..core.input import Input
from ..middlebury.input import _read_flow


class ChairsInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize)

    def _preprocess_flow(self, t, channels):
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        return tf.reshape(self._resize_crop_or_pad(t), [height, width, channels])

    def _input_flow(self):
        flow_dir = os.path.join(self.data.current_dir, 'flying_chairs/flow')
        flow_files = [os.path.join(flow_dir, fn) for fn in sorted(os.listdir(flow_dir))]

        flow, mask = _read_flow(flow_files, 1)
        flow = self._preprocess_flow(flow, 2)
        mask = self._preprocess_flow(mask, 1)
        return flow, mask

    def input_test(self):
        input_shape, im1, im2 = self._input_images('flying_chairs/test_image')
        flow, mask = self._input_flow()
        return tf.train.batch(
            [im1, im2, input_shape, flow, mask],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_raw(self, swap_images=True, shift=0):
        return super().input_raw(sequence=False,
                                 swap_images=swap_images,
                                 needs_crop=False,
                                 shift=shift)
