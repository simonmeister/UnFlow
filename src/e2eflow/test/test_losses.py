import numpy as np
import tensorflow as tf

from ..core.losses import _smoothness_deltas, create_outgoing_mask, \
    gradient_loss, compute_losses, ternary_loss

from ..core.input import read_png_image


class LossesTest(tf.test.TestCase):
    def test_smoothness_deltas(self):
        flow = np.ones([1,3,3,2], np.float32)
        flow[0, :, :, 0] = [[0,0,0],
                            [0,8,3],
                            [0,1,0]]
        flow[0, :, :, 1] = [[0,0,0],
                            [0,8,3],
                            [0,1,0]]
        delta_u_, delta_v_, mask_ = _smoothness_deltas(flow)
        delta_u_ = tf.multiply(delta_u_, mask_)
        delta_v_ = tf.multiply(delta_v_, mask_)
        sess = tf.Session()
        delta_u, delta_v, mask = sess.run([delta_u_, delta_v_, mask_])
        self.assertAllEqual(mask[0,:,:,0], [[1,1,0],
                                            [1,1,0],
                                            [1,1,0]])
        self.assertAllEqual(mask[0,:,:,1], [[1,1,1],
                                            [1,1,1],
                                            [0,0,0]])
        self.assertAllEqual(delta_u[0,:,:,0], [[0,0,0],
                                               [-8,5,0],
                                               [-1,1,0]])
        self.assertAllEqual(delta_u[0,:,:,1], [[0,-8,-3],
                                               [0,7,3],
                                               [0,0,0]])
        self.assertAllEqual(delta_v[0,:,:,0], [[0,0,0],
                                               [-8,5,0],
                                               [-1,1,0]])
        self.assertAllEqual(delta_v[0,:,:,1], [[0,-8,-3],
                                               [0,7,3],
                                               [0,0,0]])

    def test_create_outgoing_mask_all_directions(self):
       flow = np.ones([1,3,3,2], np.float32)
       flow[0, :, :, 0] = [[0,0,1],
                           [-1,3,0],
                           [0,1,0]]
       flow[0, :, :, 1] = [[-1,0,0],
                           [0,0,0],
                           [1,-1,0]]
       sess = tf.Session()
       mask = sess.run(create_outgoing_mask(flow))
       self.assertAllEqual(mask[0,:,:,0], [[0,1,0],
                                           [0,0,1],
                                           [0,1,1]])
    def test_create_outgoing_mask_large_movement(self):
       flow = np.ones([1,3,3,2], np.float32)
       flow[0, :, :, 0] = [[3,2,1],
                           [2,1,0],
                           [0,-2,-1]]
       flow[0, :, :, 1] = [[0,0,0],
                           [0,0,0],
                           [0,0,0]]
       sess = tf.Session()
       mask = sess.run(create_outgoing_mask(flow))
       self.assertAllEqual(mask[0,:,:,0], [[0,0,0],
                                           [1,1,1],
                                           [1,0,1]])

    # def test_forward_backward_loss(self):
    #     im1 = np.ones([1,3,3,3], np.float32)
    #     im2 = np.ones([1,3,3,3], np.float32)
    #     mask = np.ones([1,3,3,1], np.float32)
    #     mask[0, :, :, 0] = [[1,1,0],
    #                         [1,1,0],
    #                         [0,0,0]]
    #
    #     flow_fw = np.ones([1,3,3,2], np.float32)
    #     flow_fw[0, :, :, 0] = [[1,1,1],
    #                           [1,1,1],
    #                           [1,1,1]]
    #     flow_fw[0, :, :, 1] = [[1,1,1],
    #                           [1,1,1],
    #                           [1,1,1]]
    #     flow_bw = np.ones([1,3,3,2], np.float32)
    #     flow_bw[0, :, :, 0] = [[-1,-1,-1],
    #                           [-1,-1,-1],
    #                           [-1,-1,-1]]
    #     flow_bw[0, :, :, 1] = [[-1,-1,-1],
    #                           [-1,-1,-1],
    #                           [-1,-1,-1]]
    #
    #     sess = tf.Session()
    #     losses = sess.run(compute_losses(im1, im2, flow_fw, flow_bw, mask))
    #     self.assertAllClose(losses['fb'], 0.0, atol=1e-2)

    def test_gradient_loss(self):
        im1 = np.ones([1,3,3,3], np.float32)
        im2 = np.ones([1,3,3,3], np.float32)
        mask = np.ones([1,3,3,1], np.float32)
        im1[0, :, :, 0] = [[0,1,0],
                            [0,2,0],
                            [0,3,4]]
        im1[0, :, :, 1] = [[0,1,0],
                            [0,2,0],
                            [0,3,4]]
        im1[0, :, :, 2] = [[0,1,0],
                            [0,2,0],
                            [0,3,4]]
        im2[0, :, :, 0] = [[1,2,1],
                           [1,3,1],
                           [1,4,5]]
        im2[0, :, :, 1] = [[1,2,1],
                           [1,3,1],
                           [1,4,5]]
        im2[0, :, :, 2] = [[1,2,1],
                           [1,3,1],
                           [1,4,5]]
        sess = tf.Session()
        loss = sess.run(gradient_loss(im1, im2, mask))
        self.assertAllClose(loss, 0.0, atol=1e-2)

    def test_ternary_reference(self):
        def _ternary_reference_test(im1_name, im2_name, expected):
            with self.test_session(use_gpu=True) as sess:
                im1 = tf.expand_dims(read_png_image([im1_name]), 0)
                im2 = tf.expand_dims(read_png_image([im2_name]), 0)
                _, height, width, _ = tf.unstack(tf.shape(im1))
                mask = tf.ones([1, height, width, 1])

                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                scale = tf.cast(height * width, tf.float32)
                loss_ = ternary_loss(im1, im2, mask, max_distance=3, truncate=22) * scale
                loss = sess.run(loss_)
                print(loss)
                #self.assertAllClose(loss, expected)

        _ternary_reference_test('../test_data/frame_0011.png',
                                '../test_data/frame_0012.png',
                                8.86846e+06)
        _ternary_reference_test('../test_data/frame_0016.png',
                                '../test_data/frame_0017.png',
                                6.75537e+06)
        _ternary_reference_test('../test_data/frame_0018.png',
                                '../test_data/frame_0019.png',
                                8.22283e+06)
        _ternary_reference_test('../test_data/frame_0028.png',
                                '../test_data/frame_0029.png',
                                8.05619e+06)
