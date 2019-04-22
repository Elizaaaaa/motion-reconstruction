from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ops import keypoint_l1_loss
from .models import Encoder_resnet, Encoder_fc3_dropout
from .tf_smpl.batch_lbs import batch_rodrigues
from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot
from .util.image import unprocess_image
import time
from os.path import exists

import tensorflow as tf
import numpy as np


class Refiner(object):
    def __init__(self, config, num_frames, sess=None):
        self.config = config
        self.load_path = config.load_path
        print('checking model path {}'.format(self.load_path))
        self.num_frames = num_frames

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path

        # Visualization for fitting
        self.viz = config.viz
        self.viz_sub = 10

        # Loss & Loss weights:
        self.e_lr = config.e_lr

        self.e_loss_weight = config.e_loss_weight
        self.shape_loss_weight = config.shape_loss_weight
        self.joint_smooth_weight = config.joint_smooth_weight
        self.camera_smooth_weight = config.camera_smooth_weight
        self.keypoint_loss = keypoint_l1_loss
        self.init_pose_loss_weight = config.init_pose_loss_weight

        # Data
        self.batch_size = num_frames
        self.img_size = config.img_size

        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)
        self.img_feat_pl = tf.placeholder(tf.float32, shape=(self.batch_size, 2048))
        self.img_feat_var = tf.get_variable("img_feat_var", dtype=tf.float32, shape=(self.batch_size, 2048))
        kp_size = (self.batch_size, 19, 3)
        self.kps_pl = tf.placeholder(tf.float32, shape=kp_size)

        # Camera type!
        self.num_cam = 3
        self.proj_fn = batch_orth_proj_idrot
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)

        self.theta0_pl_shape = [self.batch_size, self.total_params]
        self.theta0_pl = tf.placeholder_with_default(
            self.load_mean_param(), shape=self.theta0_pl_shape, name='theta0')

        # Optimization space.
        self.refine_inpose = config.refine_inpose
        if self.refine_inpose:
            self.theta_pl = tf.placeholder(tf.float32, shape=self.theta0_pl_shape, name='theta_pl')
            self.theta_var = tf.get_variable("theta_var", dtype=tf.float32, shape=self.theta0_pl_shape)

        # For ft-loss
        self.shape_pl = tf.placeholder_with_default(tf.zeros(10), shape=(10,), name='beta0')
        # For stick-to-init-pose loss:
        self.init_pose_pl = tf.placeholder_with_default(tf.zeros([num_frames, 72]), shape=(num_frames, 72),
                                                        name='pose0')
        self.init_pose_weight_pl = tf.placeholder_with_default(tf.ones([num_frames, 1]), shape=(num_frames, 1),
                                                               name='pose0_weights')
        # For camera loss
        self.scale_factors_pl = tf.placeholder_with_default(tf.ones([num_frames]), shape=(num_frames),
                                                            name='scale_factors')
        self.offsets_pl = tf.placeholder_with_default(tf.zeros([num_frames, 2]), shape=(num_frames, 2), name='offsets')

