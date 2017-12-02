import tensorflow as tf
import numpy as np

from ... import ops


class DownsampleTest(tf.test.TestCase):
    def test_downsample(self):
        first = [
            [1, 1, 2, 2],
            [0, 0, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 2, 2]]
        second = [[0.5, 2], [3, 3]]

        first = np.reshape(first, [1, 4, 4, 1])
        second = np.reshape(second, [1, 2, 2, 1])

        sess = tf.Session()
        result = ops.downsample(first, 2)
        result = sess.run(result)
        self.assertAllClose(second, result)


if __name__ == "__main__":
  tf.test.main()
