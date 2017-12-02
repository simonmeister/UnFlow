import numpy as np
import tensorflow as tf

from ..core.image_warp import image_warp


class ImageWarpTest(tf.test.TestCase):
    def _warp_test(self, first, second, flow, debug=False):
        num_batch, height, width, channels = second.shape
        second_ = tf.placeholder(tf.float32, shape=second.shape, name='im')
        flow_ = tf.placeholder(tf.float32, shape=flow.shape, name='flow')
        inv_warped_second = image_warp(second_, flow_)

        sess = tf.Session()
        pred = sess.run(inv_warped_second,
            feed_dict={second_: second, flow_: flow})
        if debug:
            print('-- result channels')
            for c in range(channels):
                print(np.reshape(pred[:, :, :, c], [height, width]))
        self.assertAllClose(first, pred)

    def test_move(self):
        first = [
            [0, 0, 0, 0],
            [0, 1, 0.5, 0],
            [0, 0.3, 0.4, 0],
            [0, 0, 0, 0]]
        second = [
            [0, 1, 0, 0],
            [0, 0, 0, 0.5],
            [0.3, 0, 0, 0],
            [0, 0, 0.4, 0]]
        zero = [0, 0]
        flow = [
            [zero, [-1, 0], zero, zero],
            [zero, [0, -1], [1, 0], [0, -1]],
            [[0, -1], [-1, 0], [0, 1], zero],
            [zero, zero, [0, -1], zero]]

        self._warp_test(
            np.reshape(first, [1, 4, 4, 1]),
            np.reshape(second, [1, 4, 4, 1]),
            np.reshape(flow, [1, 4, 4, 2]))

    def test_batches(self):
        # Make sure that batches do not interfere with each other
        first_1 = [
            [0, 0, 0, 0],
            [0, 1, 0.5, 0],
            [0, 0.3, 0.4, 0],
            [0, 0, 0, 0]]
        second_1 = [
            [0, 1, 0, 0],
            [0, 0, 0, 0.5],
            [0.3, 0, 0, 0],
            [0, 0, 0.4, 0]]
        zero = [0, 0]
        flow_1 = [
            [zero, [-1, 0], zero, zero],
            [zero, [0, -1], [1, 0], [0, -1]],
            [[0, -1], [-1, 0], [0, 1], zero],
            [zero, zero, [0, -1], zero]]
        first_2 = np.zeros([4, 4])
        second_2 = np.zeros([4, 4])
        flow_2 = np.ones([4, 4, 2])
        first = np.concatenate([first_1, first_2, first_1])
        second = np.concatenate([second_1, second_2, second_1])
        flow = np.concatenate([flow_1, flow_2, flow_1])

        self._warp_test(
            np.reshape(first, [3, 4, 4, 1]),
            np.reshape(second, [3, 4, 4, 1]),
            np.reshape(flow, [3, 4, 4, 2]))

    def test_interpolate(self):
        first = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2.1]]
        second = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]]
        zero = [0, 0]
        flow = [
            [zero, zero, zero, zero],
            [zero, [-2, -2], [-2, -2], zero],
            [zero, [-2, -2], [-2, -2], zero],
            [zero, zero, zero, [-1.7, -1.6]]]

        self._warp_test(
            np.reshape(first, [1, 4, 4, 1]),
            np.reshape(second, [1, 4, 4, 1]),
            np.reshape(flow, [1, 4, 4, 2]))
