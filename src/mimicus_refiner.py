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

def shape_variance(shapes, target_shape=None):
    if target_shape is not None:
        N = tf.shape(shapes)[0]
        target_shapes = tf.tile(tf.expand_dims(target_shape, 0), [N, 1])
        return tf.losses.mean_squared_error(target_shapes, shapes)
    else:
        _, var = tf.nn.moments(shapes, axes=0)
        return tf.reduce_mean(var)

def joint_smoothness(joints):
    if joints.shape[1] == 19:
        left_hip, right_hip = 3, 2
        root = (joints[:, left_hip] + joints[:, right_hip]) / 2.
        root = tf.expand_dims(root, 1)

        joints = joints - root
    else:
        print('Unknown skeleton type')

    cur_joint = joints[:-1]
    next_joint = joints[1:]

    return tf.losses.mean_squared_error(cur_joint, next_joint)


def camera_smoothness(cams, scale_factors, offsets, img_size=224):
    scales = cams[:, 0]
    actual_scales = scales * (1./scale_factors)
    trans = cams[:, 1:]

    actual_trans = ((trans + 1) * img_size * 0.5 + offsets) / img_size

    curr_scales = actual_scales[:-1]
    next_scales = actual_scales[1:]

    curr_trans = actual_trans[:-1]
    next_trans = actual_trans[1:]

    scale_diff = tf.losses.mean_squared_error(curr_scales, next_scales)
    trans_diff = tf.losses.mean_squared_error(curr_trans, next_trans)
    return scale_diff + trans_diff


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
        self.image_feat_pl = tf.placeholder(tf.float32, shape=(self.batch_size, 2048))
        self.image_feat_var = tf.get_variable("image_feat_var", dtype=tf.float32, shape=(self.batch_size, 2048))
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

        #Build model
        self.ief = config.ief
        if self.ief:
            self.num_stage = config.num_stage
            self.build_refine_model()

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        all_vars_filtered = [v for v in all_vars if ('image_feat_var' not in v.name) and ('theta_var' not in v.name)]
        self.saver = tf.train.Saver(all_vars_filtered)

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        filtered_vars = [v for v in all_vars if ('image_feat_var' in v.name) or ('theta_var' in v.name)]
        self.sess.run(tf.variables_initializer(filtered_vars))

        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)
        self.mean_value = self.sess.run(self.mean_var)


    def predict(self, images, keypoints, scale_factors, offsets):
        print('start predicing')

        feed_dict = {self.images_pl: images}
        image_feats = self.sess.run(self.image_feat, feed_dict)

        feed_dict = {
            self.image_feat_pl: image_feats,
            self.kps_pl: keypoints,
        }

        self.sess.run(self.set_image_feat_var, feed_dict)
        if self.refine_inpose:
            use_res = -2
        else:
            use_res = -1

        fetch_dict = {
            'theta': self.final_thetas[use_res],
            'joints': self.all_keypoints[use_res],
        }

        init_result = self.sess.run(fetch_dict, feed_dict)

        shapes = init_result['theta'][:, -10:]
        mean_shape = np.mean(shapes, axis=0)
        feed_dict[self.shape_pl] = mean_shape

        init_pose = init_result['theta'][:, 3:3+72]
        feed_dict[self.init_pose_pl] = init_pose

        print('start optimization')
        if self.refine_inpose:
            print('    --------- in pose space')
            feed_dict[self.theta_pl] = init_result['theta']
            self.sess.run(self.set_theta_var, feed_dict)

        if self.config.use_weighted_init_pose:
            print('    --------- in weighted init space')
            get_keypoints = np.stack(keypoints)
            vis = get_keypoints[:, :, 2, None]
            diff = vis * (get_keypoints[:, :, :2] - init_result['joints'])
            error = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1, ord=1)
            weights = np.expand_dims(np.exp(-(error/error.max())**2), 1)

            feed_dict[self.init_pose_weight_pl] = weights

        feed_dict[self.scale_factors_pl] = scale_factors
        feed_dict[self.offsets_pl] = offsets

        #TODO: not fetch verts for now, loss_init_pose not available yet
        fetch_dict = {
            'joints': self.all_keypoints[-1],
            'cams': self.all_cams[-1],
            'joints3d': self.all_Joints[-1],
            'theta': self.final_thetas[-1],
            'total_loss': self.total_loss,
            'loss_kp': self.e_loss_kp,
            'loss_shape': self.loss_shape,
            'loss_joints': self.loss_joints,
            'loss_camera': self.loss_camera,
            'optim': self.e_opt,
        }

        #TODO: not take in loss_init_pose for now
        all_loss_keys = ['loss_kp', 'loss_shape', 'loss_joints', 'loss_camera']
        num_iter = self.config.num_refine
        loss_records = {}
        for step in range(num_iter):
            result = self.sess.run(fetch_dict, feed_dict)
            loss_keys = [key for key in all_loss_keys if key in result.keys()]
            total_loss = result['total_loss']

            #msg_prefix = 'iter %d/%d, total_loss %.2g' % (step, num_iter, total_loss)
            #msg_raw = ['%s: %.2g' % (key, result[key]) for key in loss_keys]
            #print(msg_prefix + ' ' + ' ,'.join(msg_raw))

            if step == 0:
                for key in loss_keys:
                    loss_records[key] = [result[key]]
            else:
                for key in loss_keys:
                    loss_records[key].append(result[key])

        del result['optim']

        result['loss_records'] = loss_records

        return result


    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        mean[0, 0] = 0.9

        mean = tf.constant(mean, tf.float32)

        self.mean_var = tf.Variable(mean, name="mean_param", dtype=tf.float32, trainable=True)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])

        return init_mean

    def build_refine_model(self):
        image_encoder_fn = Encoder_resnet
        thread_encoder_fn = Encoder_fc3_dropout

        self.image_feat, self.E_var = image_encoder_fn(self.images_pl, is_training=False, reuse=False)
        self.set_image_feat_var = self.image_feat_var.assign(self.image_feat_pl)

        self.all_verts = []
        self.all_keypoints = []
        self.all_cams = []
        self.all_Joints = []
        self.all_SmplJoints = []
        self.final_thetas = []

        theta_prev = self.theta0_pl

        for i in np.arange(self.num_stage):
            print('Iteration {}'.format(i))
            state = tf.concat([self.image_feat_var, theta_prev], 1)

            if i == 0:
                delta_theta, threeD_var = thread_encoder_fn(state, num_output=self.total_params, is_training=False, reuse=False)
                self.E_var.append(threeD_var)
            else:
                delta_theta, _ = thread_encoder_fn(state, num_output=self.total_params, is_training=False, reuse=True)
                theta = theta_prev + delta_theta

                cams = theta[:, :self.num_cam]
                poses = theta[:, self.num_cam:(self.num_cam + self.num_theta)]
                shapes = theta[:, (self.num_cam + self.num_theta):]

                print('check theta shape {}'.format(shapes.shape))

                #original get_skin is True, will return verts, joints and rs
                #turning off will only return joints
                joints = self.smpl(shapes, poses, get_skin=False)
                smplJoints = self.smpl.J_transformed

                pred_keypoint = self.proj_fn(joints, cams, name='proj_2d_stage%d' % i)
                self.all_keypoints.append(pred_keypoint)
                self.all_cams.append(cams)
                self.all_Joints.append(joints)
                self.all_SmplJoints.append(smplJoints)
                theta_prev = theta

        if self.refine_inpose:
            self.set_theta_var = self.theta_var.assign(self.theta_pl)
            theta_final = self.theta_var
        else:
            theta_final = theta

        cams = theta_final[:, :self.num_cam]
        poses = theta_final[:, self.num_cam:(self.num_cam + self.num_theta)]
        shapes = theta_final[:, (self.num_cam + self.num_theta):]

        joints = self.smpl(shapes, poses, get_skin=False)
        smplJoints = self.smpl.J_transformed

        pred_keypoint = self.proj_fn(joints, cams, name='proj_2d_stage%d' % (self.num_stage-1))

        self.all_keypoints.append(pred_keypoint)
        self.all_cams.append(cams)
        self.all_Joints.append(joints)
        self.all_SmplJoints.append(smplJoints)
        self.final_thetas.append(theta_final)

        # Compute new losses!!
        self.e_loss_kp = self.e_loss_weight * self.keypoint_loss(self.kps_pl,
                                                                 pred_keypoint)
        # Beta variance should be low!
        self.loss_shape = self.shape_loss_weight * shape_variance(shapes, self.shape_pl)
        # Endpoints should be smooth!!
        self.loss_joints = self.joint_smooth_weight * joint_smoothness(joints)
        # Camera should be smooth
        self.loss_camera = self.camera_smooth_weight * camera_smoothness(cams, self.scale_factors_pl, self.offsets_pl,
                                                                         img_size=self.config.img_size)

        self.total_loss = self.e_loss_kp + self.loss_shape + self.loss_joints + self.loss_camera

        print('Setting up optimizer..')
        self.optimizer = tf.train.AdamOptimizer
        e_optimizer = self.optimizer(self.e_lr)

        if self.refine_inpose:
            self.e_opt = e_optimizer.minimize(self.total_loss, var_list=[self.theta_var])
        else:
            self.e_opt = e_optimizer.minimize(self.total_loss, var_list=[self.image_feat_var])

        print('Done initializing the model!')