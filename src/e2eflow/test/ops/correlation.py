import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gradient_checker

from ... import ops


# NOTE: the tests are not exhaustive, as we assume that the published correlation code is stable

class CorrelationTest(tf.test.TestCase):
    def _test_correlation(self, in0, in1, out=None, **kwargs):
        with self.test_session(use_gpu=True) as sess:
            in0_op = tf.constant(in0, tf.float32)
            in1_op = tf.constant(in1, tf.float32)
            result_op = ops.correlation(in0_op, in1_op, **kwargs)
            result = sess.run(result_op)

            if out is not None:
                self.assertAllClose(out, result)

            jacob_t, jacob_n = gradient_checker.compute_gradient([in0_op, in1_op],
                                                                 [in0.shape, in1.shape],
                                                                 result_op, result.shape)
            #print("--------------- n")
            #print(jacob_n)
            #print("--------------- t")
            #print(jacob_t)
            self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)

    def test_correlation_trivial(self):
        first = [
            [1, 1, 2, 2],
            [0, 0, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 2, 2]]
        second = [
            [1, 1, 2, 2],
            [0, 0, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 2, 2]]

        first = np.reshape(first, [1, 1, 4, 4])
        second = np.reshape(second, [1, 1, 4, 4])
        expected = np.square(first)
        self._test_correlation(first, second, expected,
                               kernel_size=1, stride_2=1, max_displacement=0,
                               pad=0)

    def test_correlation_batch(self):
        first = [
           [1, 1, 2, 2],
           [0, 0, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 2, 2]]
        second = [
           [1, 1, 2, 2],
           [0, 0, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 2, 2]]

        first = np.reshape(first, [1, 1, 4, 4])
        second = np.reshape(second, [1, 1, 4, 4])
        expected = np.square(first)

        self._test_correlation(np.concatenate([first, first], 0),
                              np.concatenate([second, second], 0),
                              np.concatenate([expected, expected], 0),
                              kernel_size=1, stride_2=1, max_displacement=0,
                              pad=0)

    def test_correlation_channels(self):
        pass

    def test_correlation_3x3(self):
        return
        first = [
          [1, 1, 3],
          [0, 0, 1],
          [2, 2, 0.2]]
        second = [
          [1, 2, 0.1],
          [3, 4, 2.2],
          [4, 5, 1.6]]

        first = np.reshape(first, [1, 1, 3, 3])
        second = np.reshape(second, [1, 1, 3, 3])
        self._test_correlation(first, second, None,
                             kernel_size=3, stride_2=1, max_displacement=1,
                             pad=2)

if __name__ == "__main__":
  tf.test.main()
