import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gradient_checker

from ... import ops


class ForwardWarpTest(tf.test.TestCase):
    def test_grad(self):
        with self.test_session(use_gpu=True) as sess:
            flow_shape = [1, 10, 10, 2]
            warped_shape = [1, 10, 10, 1]

            flow_ = tf.placeholder(tf.float32, shape=flow_shape, name='flow')
            warped_ = ops.forward_warp(flow_)

            jacob_t, jacob_n = gradient_checker.compute_gradient(flow_, flow_shape,
                                                                 warped_, warped_shape)
            self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)


if __name__ == "__main__":
  tf.test.main()
