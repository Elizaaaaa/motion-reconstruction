from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ops import keypoint_l1_loss
from .models import Encoder_resnet, Encoder_fc3_dropout
from .tf_smpl.batch_lbs import batch_rodrigues
from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot
from .util.renderer import SMPLRenderer, draw_skeleton
from .util.image import unprocess_image
import time
from os.path import exists

import tensorflow as tf
import numpy as np


class Refiner(object):
    def __init__(self, config, num_frames, sess=None):
        return 0
