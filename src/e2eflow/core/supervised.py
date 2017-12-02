import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .augment import random_photometric
from .flow_util import flow_to_color
from .losses import charbonnier_loss
from .flownet import flownet
from .unsupervised import _track_image, _track_loss, FLOW_SCALE


def supervised_loss(batch, params, normalization=None):
    channel_mean = tf.constant(normalization[0]) / 255.0
    im1, im2, flow_gt, mask_gt = batch
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im_shape = tf.shape(im1)[1:3]

    # -------------------------------------------------------------------------

    im1_photo, im2_photo = random_photometric(
        [im1, im2],
        noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
        brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
        min_gamma=0.7, max_gamma=1.5)

    _track_image(im1_photo, 'im1_photo')
    _track_image(im2_photo, 'im2_photo')
    _track_image(flow_to_color(flow_gt), 'flow_gt')
    _track_image(mask_gt, 'mask_gt')

    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    flownet_spec = params.get('flownet', 'S')
    full_resolution = params.get('full_res')
    train_all = params.get('train_all')
    # -------------------------------------------------------------------------
    # FlowNet
    flows_fw = flownet(im1_photo, im2_photo,
                       flownet_spec=flownet_spec,
                       full_resolution=full_resolution,
                       train_all=train_all)
    
    if not train_all:
        flows_fw = [flows_fw[-1]]
    final_loss = 0.0
    for i, net_flows in enumerate(reversed(flows_fw)):
        flow_fw = net_flows[0]
        if params.get('full_res'):
            final_flow_fw = flow_fw * FLOW_SCALE * 4
        else:
            final_flow_fw = tf.image.resize_bilinear(flow_fw, im_shape) * FLOW_SCALE * 4
        _track_image(flow_to_color(final_flow_fw), 'flow_pred_' + str(i))

        net_loss = charbonnier_loss(final_flow_fw - flow_gt, mask_gt)
        final_loss += net_loss / (2 ** i)

    regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
    final_loss += regularization_loss
    _track_loss(regularization_loss, 'loss/regularization')
    _track_loss(final_loss, 'loss/combined')

    return final_loss
