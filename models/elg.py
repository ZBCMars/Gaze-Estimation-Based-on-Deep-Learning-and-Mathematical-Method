"""ELG architecture."""
from typing import Dict

import numpy as np
import scipy
import tensorflow as tf
import scipy.optimize
#import tensorflow_probability as tfp
from models import newton

from core import BaseDataSource, BaseModel


def _tf_mse(x, y):
    """Tensorflow call for mean-squared error."""
    return tf.reduce_mean(tf.squared_difference(x, y))


class ELG(BaseModel):
    """ELG architecture as introduced in [Park et al. ETRA'18]."""

    def __init__(self, tensorflow_session=None, first_layer_stride=1,
                 num_modules=2, num_feature_maps=32, **kwargs):
        """Specify ELG-specific parameters."""
        self._hg_first_layer_stride = first_layer_stride
        self._hg_num_modules = num_modules
        self._hg_num_feature_maps= num_feature_maps

        # Call parent class constructor
        super().__init__(tensorflow_session, **kwargs)

    _hg_first_layer_stride = 1
    _hg_num_modules = 2
    _hg_num_feature_maps = 32
    _hg_num_landmarks = 18
    _hg_num_residual_blocks = 1

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        first_data_source = next(iter(self._train_data.values()))
        input_tensors = first_data_source.output_tensors
        if self._data_format == 'NHWC':
            _, eh, ew, _ = input_tensors['eye'].shape.as_list()
        else:
            _, _, eh, ew = input_tensors['eye'].shape.as_list()
        return 'ELG_i%dx%d_f%dx%d_n%d_m%d' % (
            ew, eh,
            int(ew / self._hg_first_layer_stride),
            int(eh / self._hg_first_layer_stride),
            self._hg_num_feature_maps, self._hg_num_modules,
        )

    def train_loop_pre(self, current_step):
        """Run this at beginning of training loop."""
        # Set difficulty of training data
        data_source = next(iter(self._train_data.values()))
        data_source.set_difficulty(min((1. / 1e6) * current_step, 1.))

    _column_of_ones = None
    _column_of_zeros = None

    def _augment_training_images(self, images, mode):
        if mode == 'test':
        	   print(images)
        	   images = tf.transpose(images, perm=[0, 2, 3, 1])
        	   print(images)
        	   images = tf.image.resize_images(images, [36,60],method=0)
        	   print(images)
        	   images = tf.transpose(images, perm=[0, 3, 1, 2])
        	   print(images)
        	   '''
        	   images = tf.image.resize_images(images, [36,60],method=0)
        	   print(images)
        	   '''
        	   return images
        with tf.variable_scope('augment'):
            if self._data_format == 'NCHW':
                images = tf.transpose(images, perm=[0, 2, 3, 1])
            n, h, w, _ = images.shape.as_list()
            if self._column_of_ones is None:
                self._column_of_ones = tf.ones((n, 1))
                self._column_of_zeros = tf.zeros((n, 1))
            transforms = tf.concat([
                self._column_of_ones,
                self._column_of_zeros,
                tf.truncated_normal((n, 1), mean=0, stddev=.05*w),
                self._column_of_zeros,
                self._column_of_ones,
                tf.truncated_normal((n, 1), mean=0, stddev=.05*h),
                self._column_of_zeros,
                self._column_of_zeros,
            ], axis=1)
            images = tf.contrib.image.transform(images, transforms, interpolation='BILINEAR')
            if self._data_format == 'NCHW':
                images = tf.transpose(images, perm=[0, 3, 1, 2])
        return images

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        print()
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        if mode=='train':
            print(x)
        y1 = input_tensors['heatmaps'] if 'heatmaps' in input_tensors else None
        y2 = input_tensors['landmarks'] if 'landmarks' in input_tensors else None
        y3 = input_tensors['radius'] if 'radius' in input_tensors else None

        y4 = input_tensors['gazemaps'] if 'gazemaps' in input_tensors else None
        y5 = input_tensors['gaze'] if 'gaze' in input_tensors else None

        with tf.variable_scope('input_data'):
            self.summary.feature_maps('eyes', x, data_format=self._data_format_longer)
            if y1 is not None:
                self.summary.feature_maps('hmaps_true', y1, data_format=self._data_format_longer)

        x = self._augment_training_images(x, mode)
        
        outputs = {}
        loss_terms = {}
        metrics = {}

        feature_map = self.Base_model(x)
        h1, h2 = self.Stage_1(feature_map)
        for i in range(5):
            h1, h2 = self.Stage_x(tf.concat([h1, h2, feature_map], axis = 1))
        '''
        h1, h2 = self.Stage_x(tf.concat([h1, h2, feature_map], axis = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.Stage_x(tf.concat([h1, h2, feature_map], axis = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.Stage_x(tf.concat([h1, h2, feature_map], axis = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.Stage_x(tf.concat([h1, h2, feature_map], axis = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        '''

        with tf.variable_scope('hourglass'):
            # TODO: Find better way to specify no. landmarks
            if y1 is not None:
                if self._data_format == 'NCHW':
                    self._hg_num_landmarks = y1.shape.as_list()[1]
                if self._data_format == 'NHWC':
                    self._hg_num_landmarks = y1.shape.as_list()[3]
            else:
                self._hg_num_landmarks = 18
            assert self._hg_num_landmarks == 18

            # Prepare for Hourglass by downscaling via conv
            with tf.variable_scope('pre'):
                n = self._hg_num_feature_maps
                x = self._apply_conv(x, num_features=n, kernel_size=7,
                                     stride=self._hg_first_layer_stride)
                x = tf.nn.relu(self._apply_bn(x))
                x = self._build_residual_block(x, n, 2*n, name='res1')
                x = self._build_residual_block(x, 2*n, n, name='res2')

            # Hourglass blocks
            x_prev = x
            for i in range(self._hg_num_modules):
                with tf.variable_scope('hg_%d' % (i + 1)):
                    x = self._build_hourglass(x, steps_to_go=4, num_features=self._hg_num_feature_maps)
                    x, h = self._build_hourglass_after(
                        x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
                    )
                    self.summary.feature_maps('hmap%d' % i, h, data_format=self._data_format_longer)
                    if y1 is not None:
                        metrics['heatmap%d_mse' % (i + 1)] = _tf_mse(h, y1)
                    x_prev = x
            if y1 is not None:
                loss_terms['heatmaps_mse'] = tf.reduce_mean([
                    metrics['heatmap%d_mse' % (i + 1)] for i in range(self._hg_num_modules)
                ])
            x = h
            '''print(x)
                                                print(h1)
                                                print(h2)'''
            zzz = x
            for i in range(3):
                zzz = self._apply_pool(zzz, kernel_size = 2, stride = 2)
                print(zzz)
            outputs['heatmaps'] =  0.5*zzz + 0.5*h2 #x
            #heatmaps_train = outputs['heatmaps'].eval()

        # Soft-argmax
        #print(np.shape(x))


        

        x = self._calculate_landmarks(x)

        with tf.variable_scope('upscale'):
            # Upscale since heatmaps are half-scale of original image
            x *= self._hg_first_layer_stride
            if y2 is not None:
                metrics['landmarks_mse'] = _tf_mse(x, y2)
            outputs['landmarks'] = x

        # Fully-connected layers for radius regression
        with tf.variable_scope('radius'):
            x = tf.contrib.layers.flatten(tf.transpose(x, perm=[0, 2, 1]))
            for i in range(3):
                with tf.variable_scope('fc%d' % (i + 1)):
                    x = tf.nn.relu(self._apply_bn(self._apply_fc(x, 100)))
            with tf.variable_scope('out'):
                x = self._apply_fc(x, 1)
            outputs['radius'] = x
            if y3 is not None:
                metrics['radius_mse'] = _tf_mse(tf.reshape(x, [-1]), y3)
                loss_terms['radius_mse'] = 1e-7 * metrics['radius_mse']
            self.summary.histogram('radius', x)


        '''heatmaps_amax = np.amax(heatmaps_train.reshape(-1, 18), axis=0)
                                can_use_eye = np.all(heatmaps_amax > 0.7)
                                can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
                                can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)'''

        #with tf.Session() as sess1:

        batch_size = 32

        for j in range(batch_size):

            eyes = []

            eye_landmarks = outputs['landmarks'][j, : ]

            #print(eye_landmarks)
            eye_radius = (outputs['radius'])[j][0]


            '''
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                inv_centre_mat)

            eyes.append({
                #'image': eye_image,
                'inv_landmarks_transform_mat': inv_transform_mat,
                #'side': 'left' if is_left else 'right',
            })'''
            
            #if eye_side == 'left':
                #eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                #eye_image = np.fliplr(eye_image)

            #face_index = int(eye_index / 2)
            #eh, ew, _ = eye_image_raw.shape
            #v0 = face_index * 2 * eh
            #v1 = v0 + eh
            #v2 = v1 + eh
            #u0 = 0 if eye_side == 'left' else ew
            #u1 = u0 + ew
            #bgr[v0:v1, u0:u1] = eye_image_raw
            #bgr[v1:v2, u0:u1] = eye_image_annotated

            # Transform predictions
            '''
            eye_landmarks = np.concatenate([eye_landmarks,
                                            [[eye_landmarks[-1, 0] + eye_radius,
                                              eye_landmarks[-1, 1]]]])'''
            #print(eye_landmarks)
            #print([[eye_landmarks[-1, 0] + eye_radius,
                                              #eye_landmarks[-1, 1]]])
            eye_landmarks = tf.concat([eye_landmarks,
                                            [[eye_landmarks[-1, 0] + eye_radius,
                                              eye_landmarks[-1, 1]]]], 0)
            #print(eye_landmarks)
            '''
            Interpret the input as a matrix.
            eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                               'constant', constant_values=1.0))
            '''
            eye_landmarks = tf.pad(eye_landmarks, ((0, 0), (0, 1)),
                                  						'constant', constant_values=1.0)
            #print(eye_landmarks)
            #eye_landmarks = (eye_landmarks *
                             #eye['inv_landmarks_transform_mat'].T)[:, :2]

            eye_landmarks = eye_landmarks[:, :2]
            #eye_landmarks = np.asarray(eye_landmarks)   将结构数据转换为ndarray类型
            eye_landmarks = eye_landmarks
            #print(eye_landmarks)
            eyelid_landmarks = eye_landmarks[0:8, :]
            #print(eyelid_landmarks)
            iris_landmarks = eye_landmarks[8:16, :]
            #print(iris_landmarks)
            iris_centre = eye_landmarks[16, :]
            #print(iris_centre)
            eyeball_centre = eye_landmarks[17, :]
            #print(eyeball_centre)
            #eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
            #                                eye_landmarks[17, :])
            eyeball_radius = tf.norm(eye_landmarks[18, :] -
                                            eye_landmarks[17, :])





            x = estimate_gaze_from_landmarks(iris_landmarks, iris_centre, eyeball_centre, eyeball_radius,
                                     initial_gaze=None)

        '''with tf.variable_scope('exchange2'):
                                    x = tf.convert_to_tensor(x)'''

        if y4 is not None:
            # Cross-entropy loss
            metrics['gazemaps_ce'] = -tf.reduce_mean(tf.reduce_sum(
                y4 * tf.log(tf.clip_by_value(gmap, 1e-10, 1.0)),  # avoid NaN
                axis=[1, 2, 3]))

        if y5 is not None:
            metrics['gaze_mse'] = tf.reduce_mean(tf.squared_difference(x, y5))
            metrics['gaze_ang'] = tensorflow_angular_error_from_pitchyaw(y5, x)

        if y4 is not None and y5 is not None:
            loss_terms['combined_loss'] = 1e-5*metrics['gazemaps_ce'] + metrics['gaze_mse']

        # Define outputs
        return outputs, loss_terms, metrics

    def tensorflow_angular_error_from_pitchyaw(y_true, y_pred):
        def angles_to_unit_vectors(y):
            sin = tf.sin(y)
            cos = tf.cos(y)
            return tf.stack([
                tf.multiply(cos[:, 0], sin[:, 1]),
                sin[:, 0],
                tf.multiply(cos[:, 0], cos[:, 1]),
            ], axis=1)

        with tf.name_scope('mean_angular_error'):
            v_true = angles_to_unit_vectors(y_true)
            v_pred = angles_to_unit_vectors(y_pred)
            return tensorflow_angular_error_from_vector(v_true, v_pred)
		
    def _apply_conv(self, tensor, num_features, kernel_size=3, stride=1):
        return tf.layers.conv2d(
            tensor,
            num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            data_format=self._data_format_longer,
            name='conv',
        )

    def _apply_fc(self, tensor, num_outputs):
        return tf.layers.dense(
            tensor,
            num_outputs,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            name='fc',
        )

    def _apply_pool(self, tensor, kernel_size=3, stride=2):
        tensor = tf.layers.max_pooling2d(
            tensor,
            pool_size=kernel_size,
            strides=stride,
            padding='SAME',
            data_format=self._data_format_longer,
            name='pool',
        )
        return tensor

    def _apply_bn(self, tensor):
        return tf.contrib.layers.batch_norm(
            tensor,
            scale=True,
            center=True,
            is_training=self.use_batch_statistics,
            trainable=True,
            data_format=self._data_format,
            updates_collections=None,
        )

    def _build_residual_block(self, x, num_in, num_out, name='res_block'):
        with tf.variable_scope(name):
            half_num_out = max(int(num_out/2), 1)
            c = x
            with tf.variable_scope('conv1'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=1, stride=1)
            with tf.variable_scope('conv2'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=3, stride=1)
            with tf.variable_scope('conv3'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=num_out, kernel_size=1, stride=1)
            with tf.variable_scope('skip'):
                if num_in == num_out:
                    s = tf.identity(x)
                else:
                    s = self._apply_conv(x, num_features=num_out, kernel_size=1, stride=1)
            x = c + s
        return x

    def _build_hourglass(self, x, steps_to_go, num_features, depth=1):
        with tf.variable_scope('depth%d' % depth):
            # Upper branch
            up1 = x
            for i in range(self._hg_num_residual_blocks):
                up1 = self._build_residual_block(up1, num_features, num_features,
                                                 name='up1_%d' % (i + 1))
            # Lower branch
            low1 = self._apply_pool(x, kernel_size=2, stride=2)
            for i in range(self._hg_num_residual_blocks):
                low1 = self._build_residual_block(low1, num_features, num_features,
                                                  name='low1_%d' % (i + 1))
            # Recursive
            low2 = None
            if steps_to_go > 1:
                low2 = self._build_hourglass(low1, steps_to_go - 1, num_features, depth=depth+1)
            else:
                low2 = low1
                for i in range(self._hg_num_residual_blocks):
                    low2 = self._build_residual_block(low2, num_features, num_features,
                                                      name='low2_%d' % (i + 1))
            # Additional residual blocks
            low3 = low2
            for i in range(self._hg_num_residual_blocks):
                low3 = self._build_residual_block(low3, num_features, num_features,
                                                  name='low3_%d' % (i + 1))
            # Upsample
            if self._data_format == 'NCHW':  # convert to NHWC
                low3 = tf.transpose(low3, (0, 2, 3, 1))
            up2 = tf.image.resize_bilinear(
                    low3,
                    up1.shape[1:3] if self._data_format == 'NHWC' else up1.shape[2:4],
                    align_corners=True,
                  )
            if self._data_format == 'NCHW':  # convert back from NHWC
                up2 = tf.transpose(up2, (0, 3, 1, 2))

        return up1 + up2

    def _build_hourglass_after(self, x_prev, x_now, do_merge=True):
        with tf.variable_scope('after'):
            for j in range(self._hg_num_residual_blocks):
                x_now = self._build_residual_block(x_now, self._hg_num_feature_maps,
                                                   self._hg_num_feature_maps,
                                                   name='after_hg_%d' % (j + 1))
            x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
            x_now = self._apply_bn(x_now)
            x_now = tf.nn.relu(x_now)

            with tf.variable_scope('hmap'):
                h = self._apply_conv(x_now, self._hg_num_landmarks, kernel_size=1, stride=1)

        x_next = x_now
        if do_merge:
            with tf.variable_scope('merge'):
                with tf.variable_scope('h'):
                    x_hmaps = self._apply_conv(h, self._hg_num_feature_maps, kernel_size=1, stride=1)
                with tf.variable_scope('x'):
                    x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
                x_next += x_prev + x_hmaps
        return x_next, h

    _softargmax_coords = None

    def _calculate_landmarks(self, x):
        """Estimate landmark location from heatmaps."""
        with tf.variable_scope('argsoftmax'):
            if self._data_format == 'NHWC':
                _, h, w, _ = x.shape.as_list()
                #_, h, w, c = x.shape.as_list()
            else:
                _, _, h, w = x.shape.as_list()
                #_, c, h, w = x.shape.as_list()
            if self._softargmax_coords is None:
                # Assume normalized coordinate [0, 1] for numeric stability
                ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                             np.linspace(0, 1.0, num=h, endpoint=True),
                                             indexing='xy')
                #print(np.shape(ref_xs))
                ref_xs = np.reshape(ref_xs, [-1, 1, h*w])
                ref_ys = np.reshape(ref_ys, [-1, 1, h*w])
                #print(np.shape(ref_xs))
                self._softargmax_coords = (
                    tf.constant(ref_xs, dtype=tf.float32),
                    tf.constant(ref_ys, dtype=tf.float32),
                )
            ref_xs, ref_ys = self._softargmax_coords

            # Assuming N x 18 x 45 x 75 (NCHW)
            beta = 1e2
            if self._data_format == 'NHWC':
                x = tf.transpose(x, (0, 3, 1, 2))
            #print(h)
            #print(w)
            x = tf.reshape(x, [-1, self._hg_num_landmarks, h*w])
            #x = tf.reshape(x, [-1, h*w])
            x = tf.nn.softmax(beta * x, axis=-1)

            #print(np.shape(ref_xs))
            #print(np.shape(x))

            lmrk_xs = tf.reduce_sum(ref_xs * x, axis=[2])
            lmrk_ys = tf.reduce_sum(ref_ys * x, axis=[2])

            #lmrk_xs = tf.reduce_sum(ref_xs * x, axis=[1])
            #lmrk_ys = tf.reduce_sum(ref_ys * x, axis=[1])

            # Return to actual coordinates ranges
            return tf.stack([
                lmrk_xs * (w - 1.0) + 0.5,
                lmrk_ys * (h - 1.0) + 0.5,
            ], axis=2)  # N x 18 x 2

    def VGG_Base(self, x):

        with tf.variable_scope('VGG_Base', reuse=tf.AUTO_REUSE):
            c = x
            with tf.variable_scope('VGG_conv1_1', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=64, kernel_size=3, stride=1)
            with tf.variable_scope('VGG_conv1_2', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=64, kernel_size=3, stride=1)
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_pool(c, kernel_size = 2, stride = 2)
            with tf.variable_scope('VGG_conv2_1', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=128, kernel_size=3, stride=1)
            with tf.variable_scope('VGG_conv2_2', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=128, kernel_size=3, stride=1)
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_pool(c, kernel_size = 2, stride = 2)
            with tf.variable_scope('VGG_conv3_1', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=256, kernel_size=3, stride=1)
            with tf.variable_scope('VGG_conv3_2', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=256, kernel_size=3, stride=1)
            with tf.variable_scope('VGG_conv3_3', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=256, kernel_size=3, stride=1)
            with tf.variable_scope('VGG_conv3_4', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=256, kernel_size=3, stride=1)
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_pool(c, kernel_size = 2, stride = 2)
            with tf.variable_scope('VGG_conv4_1', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=512, kernel_size=3, stride=1)
            with tf.variable_scope('VGG_conv4_2', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=512, kernel_size=3, stride=1)
                c = tf.nn.relu(self._apply_bn(c))

        return c

    def Base_model(self, x):
        
        with tf.variable_scope('Base_model', reuse=tf.AUTO_REUSE):
            c = x
            with tf.variable_scope('Base_conv1', reuse=tf.AUTO_REUSE):
                c = self.VGG_Base(c)
            with tf.variable_scope('Base_conv2', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=256, kernel_size=3, stride=1)
            with tf.variable_scope('Base_conv3', reuse=tf.AUTO_REUSE):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=128, kernel_size=3, stride=1)
                c = tf.nn.relu(self._apply_bn(c))
        return c
        

    def Stage_1(self, x):

        with tf.variable_scope('Stage_1', reuse=tf.AUTO_REUSE):
            c = x
            with tf.variable_scope('L1_conv1', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c))
                c1 = self._apply_conv(c1, num_features=128, kernel_size=3, stride=1)
            with tf.variable_scope('L1_conv2', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features=128, kernel_size=3, stride=1)
            with tf.variable_scope('L1_conv3', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features=128, kernel_size=3, stride=1)
            with tf.variable_scope('L1_conv4', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features=512, kernel_size=1, stride=1)
            with tf.variable_scope('L1_conv5', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features=36, kernel_size=1, stride=1)



            with tf.variable_scope('L2_conv1', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c))
                c2 = self._apply_conv(c2, num_features=128, kernel_size=3, stride=1)
            with tf.variable_scope('L2_conv2', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features=128, kernel_size=3, stride=1)
            with tf.variable_scope('L2_conv3', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features=128, kernel_size=3, stride=1)
            with tf.variable_scope('L2_conv4', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features=512, kernel_size=1, stride=1)
            with tf.variable_scope('L2_conv5', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features=18, kernel_size=1, stride=1)
            
            

        return c1, c2
    	
    def Stage_x(self, x):

        with tf.variable_scope('Stage_x', reuse=tf.AUTO_REUSE):
            c = x
            with tf.variable_scope('L1_conv1', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c))
                c1 = self._apply_conv(c1, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L1_conv2', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L1_conv3', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L1_conv4', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L1_conv5', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L1_conv6', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L1_conv7', reuse=tf.AUTO_REUSE):
                c1 = tf.nn.relu(self._apply_bn(c1))
                c1 = self._apply_conv(c1, num_features = 36, kernel_size = 7, stride = 1)
           

            with tf.variable_scope('L2_conv1', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c))
                c2 = self._apply_conv(c2, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L2_conv2', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L2_conv3', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L2_conv4', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L2_conv5', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L2_conv6', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features = 128, kernel_size = 7, stride = 1)
            with tf.variable_scope('L2_conv7', reuse=tf.AUTO_REUSE):
                c2 = tf.nn.relu(self._apply_bn(c2))
                c2 = self._apply_conv(c2, num_features = 18, kernel_size = 7, stride = 1)
                  
        return c1, c2

def estimate_gaze_from_landmarks(iris_landmarks, iris_centre, eyeball_centre, eyeball_radius,
                                 initial_gaze=None):
    """Given iris edge landmarks and other coordinates, estimate gaze direction.

    More correctly stated, estimate gaze from iris edge landmark coordinates, iris centre
    coordinates, eyeball centre coordinates, and eyeball radius in pixels.
    """
    #e_x0, e_y0 = eyeball_centre
    #i_x0, i_y0 = iris_centre
    e_x0 = tf.slice(eyeball_centre,[0],[1])
    #print(e_x0)
    e_y0 = tf.slice(eyeball_centre,[1],[1])
    #print(e_y0)

    i_x0 = tf.slice(iris_centre,[0],[1])
    #print(i_x0)
    i_y0 = tf.slice(iris_centre,[1],[1])

    if initial_gaze is not None:
        theta, phi = initial_gaze
        # theta = -theta
    else:
        #theta = np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
        #theta = tf.asin(tf.clip_by_value(tf.div(tf.subtract(i_y0, e_y0), tf.constant(eyeball_radius)), clip_value_min=-1.0, clip_value_max=1.0))
        theta = tf.asin(tf.clip_by_value(tf.div(tf.subtract(i_y0, e_y0), eyeball_radius), clip_value_min=-1.0, clip_value_max=1.0))
        
        #phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))
        phi = tf.asin(tf.clip_by_value(tf.div(tf.subtract(i_x0, e_x0), tf.multiply(eyeball_radius, -tf.cos(theta))), clip_value_min=-1.0, clip_value_max=1.0))

    delta = 0.1 * np.pi


    '''
    if iris_landmarks[0, 0] < iris_landmarks[4, 0]:  # flipped
    				#alphas = np.flip(np.arange(0.0, 2.0 * np.pi, step=np.pi/4.0), axis=0)
        alphas = tf.image.flip_up_down(tf.range(0.0, 2.0 * np.pi, step=np.pi/4.0))
        tf.reverse(tf.constant(list(np.arange(0.0, 2.0 * np.pi, step=np.pi/4.0))),axis=0)
    else:
    				#alphas = np.arange(-np.pi, np.pi, step=np.pi/4.0) + np.pi/4.0
        alphas = tf.range(-np.pi, np.pi, step=np.pi/4.0) + np.pi/4.0
    '''

    #print("2333333333:", tf.constant(list(np.arange(0.0, 2.0 * np.pi, step=np.pi/4.0))))
    z_to_pi = []
    i = 2.0 * np.pi
    while i:
        z_to_pi.append(i)
        i -= np.pi/4.0
    z_to_pi.append(0)
    #print("piiiiiiiiiiiiiiiiii:", z_to_pi)


    alphas  = tf.cond(
        iris_landmarks[0, 0] < iris_landmarks[4, 0],
        	lambda: tf.constant(z_to_pi),
        	lambda: tf.range(-np.pi, np.pi, delta=np.pi/4.0) + np.pi/4.0)

    #sin_alphas = np.sin(alphas)
    sin_alphas = tf.sin(alphas)

    #cos_alphas = np.cos(alphas)
    cos_alphas = tf.cos(alphas)

    '''
    def gaze_fit_loss_func_np(inputs):
        theta, phi, delta, phase = inputs
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        # sin_alphas_shifted = np.sin(alphas + phase)
        sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
        # cos_alphas_shifted = np.cos(alphas + phase)
        cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_delta_sin = np.sin(delta * sin_alphas_shifted)
        sin_delta_cos = np.sin(delta * cos_alphas_shifted)
        cos_delta_sin = np.cos(delta * sin_alphas_shifted)
        cos_delta_cos = np.cos(delta * cos_alphas_shifted)
        # x = -np.cos(theta + delta * sin_alphas_shifted)
        x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
        # x *= np.sin(phi + delta * cos_alphas_shifted)
        x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
        x = x1 * x2
        # y = np.sin(theta + delta * sin_alphas_shifted)
        y1 = sin_theta * cos_delta_sin
        y2 = cos_theta * sin_delta_sin
        y = y1 + y2

        ix = e_x0 + eyeball_radius * x
        iy = e_y0 + eyeball_radius * y
        dx = ix - iris_landmarks[:, 0]
        dy = iy - iris_landmarks[:, 1]
        out = np.mean(dx ** 2 + dy ** 2)

        # In addition, match estimated and actual iris centre
        iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
        iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
        out += iris_dx ** 2 + iris_dy ** 2

        # sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
        # cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase
        dsin_alphas_shifted_dphase = -sin_alphas * sin_phase + cos_alphas * cos_phase
        dcos_alphas_shifted_dphase = -cos_alphas * sin_phase - sin_alphas * cos_phase

        # sin_delta_sin = np.sin(delta * sin_alphas_shifted)
        # sin_delta_cos = np.sin(delta * cos_alphas_shifted)
        # cos_delta_sin = np.cos(delta * sin_alphas_shifted)
        # cos_delta_cos = np.cos(delta * cos_alphas_shifted)
        dsin_delta_sin_ddelta = cos_delta_sin * sin_alphas_shifted
        dsin_delta_cos_ddelta = cos_delta_cos * cos_alphas_shifted
        dcos_delta_sin_ddelta = -sin_delta_sin * sin_alphas_shifted
        dcos_delta_cos_ddelta = -sin_delta_cos * cos_alphas_shifted
        dsin_delta_sin_dphase = cos_delta_sin * delta * dsin_alphas_shifted_dphase
        dsin_delta_cos_dphase = cos_delta_cos * delta * dcos_alphas_shifted_dphase
        dcos_delta_sin_dphase = -sin_delta_sin * delta * dsin_alphas_shifted_dphase
        dcos_delta_cos_dphase = -sin_delta_cos * delta * dcos_alphas_shifted_dphase

        # x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
        # x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
        dx1_dtheta = sin_theta * cos_delta_sin + cos_theta * sin_delta_sin
        dx2_dtheta = 0.0
        dx1_dphi = 0.0
        dx2_dphi = cos_phi * cos_delta_cos - sin_phi * sin_delta_cos
        dx1_ddelta = -cos_theta * dcos_delta_sin_ddelta + sin_theta * dsin_delta_sin_ddelta
        dx2_ddelta = sin_phi * dcos_delta_cos_ddelta + cos_phi * dsin_delta_cos_ddelta
        dx1_dphase = -cos_theta * dcos_delta_sin_dphase + sin_theta * dsin_delta_sin_dphase
        dx2_dphase = sin_phi * dcos_delta_cos_dphase + cos_phi * dsin_delta_cos_dphase

        # y1 = sin_theta * cos_delta_sin
        # y2 = cos_theta * sin_delta_sin
        dy1_dtheta = cos_theta * cos_delta_sin
        dy2_dtheta = -sin_theta * sin_delta_sin
        dy1_dphi = 0.0
        dy2_dphi = 0.0
        dy1_ddelta = sin_theta * dcos_delta_sin_ddelta
        dy2_ddelta = cos_theta * dsin_delta_sin_ddelta
        dy1_dphase = sin_theta * dcos_delta_sin_dphase
        dy2_dphase = cos_theta * dsin_delta_sin_dphase

        # x = x1 * x2
        # y = y1 + y2
        dx_dtheta = dx1_dtheta * x2 + x1 * dx2_dtheta
        dx_dphi = dx1_dphi * x2 + x1 * dx2_dphi
        dx_ddelta = dx1_ddelta * x2 + x1 * dx2_ddelta
        dx_dphase = dx1_dphase * x2 + x1 * dx2_dphase
        dy_dtheta = dy1_dtheta + dy2_dtheta
        dy_dphi = dy1_dphi + dy2_dphi
        dy_ddelta = dy1_ddelta + dy2_ddelta
        dy_dphase = dy1_dphase + dy2_dphase

        # ix = w_2 + eyeball_radius * x
        # iy = h_2 + eyeball_radius * y
        dix_dtheta = eyeball_radius * dx_dtheta
        dix_dphi = eyeball_radius * dx_dphi
        dix_ddelta = eyeball_radius * dx_ddelta
        dix_dphase = eyeball_radius * dx_dphase
        diy_dtheta = eyeball_radius * dy_dtheta
        diy_dphi = eyeball_radius * dy_dphi
        diy_ddelta = eyeball_radius * dy_ddelta
        diy_dphase = eyeball_radius * dy_dphase

        # dx = ix - iris_landmarks[:, 0]
        # dy = iy - iris_landmarks[:, 1]
        ddx_dtheta = dix_dtheta
        ddx_dphi = dix_dphi
        ddx_ddelta = dix_ddelta
        ddx_dphase = dix_dphase
        ddy_dtheta = diy_dtheta
        ddy_dphi = diy_dphi
        ddy_ddelta = diy_ddelta
        ddy_dphase = diy_dphase

        # out = dx ** 2 + dy ** 2
        dout_dtheta = np.mean(2 * (dx * ddx_dtheta + dy * ddy_dtheta))
        dout_dphi = np.mean(2 * (dx * ddx_dphi + dy * ddy_dphi))
        dout_ddelta = np.mean(2 * (dx * ddx_ddelta + dy * ddy_ddelta))
        dout_dphase = np.mean(2 * (dx * ddx_dphase + dy * ddy_dphase))

        # iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
        # iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
        # out += iris_dx ** 2 + iris_dy ** 2
        dout_dtheta += 2 * eyeball_radius * (sin_theta * sin_phi * iris_dx + cos_theta * iris_dy)
        dout_dphi += 2 * eyeball_radius * (-cos_theta * cos_phi * iris_dx)

        return out, np.array([dout_dtheta, dout_dphi, dout_ddelta, dout_dphase])
        '''

    def gaze_fit_loss_func_tf(inputs):
        theta, phi, delta, phase = inputs
        sin_phase = tf.sin(phase)
        cos_phase = tf.cos(phase)
        # sin_alphas_shifted = np.sin(alphas + phase)
        sin_alphas_shifted = tf.multiply(sin_alphas, cos_phase) + tf.multiply(cos_alphas, sin_phase)
        # cos_alphas_shifted = np.cos(alphas + phase)
        cos_alphas_shifted = tf.subtract(tf.multiply(cos_alphas, cos_phase), tf.multiply(sin_alphas, sin_phase))

        sin_theta = tf.sin(theta)
        cos_theta = tf.cos(theta)
        sin_phi = tf.sin(phi)
        cos_phi = tf.cos(phi)
        sin_delta_sin = tf.sin(tf.multiply(delta, sin_alphas_shifted))
        sin_delta_cos = tf.sin(tf.multiply(delta, cos_alphas_shifted))
        cos_delta_sin = tf.cos(tf.multiply(delta, sin_alphas_shifted))
        cos_delta_cos = tf.cos(tf.multiply(delta, cos_alphas_shifted))
        # x = -np.cos(theta + delta * sin_alphas_shifted)
        x1 = tf.multiply(-cos_theta, cos_delta_sin) + tf.multiply(sin_theta, sin_delta_sin)
        # x *= np.sin(phi + delta * cos_alphas_shifted)
        x2 = tf.multiply(sin_phi, cos_delta_cos) + tf.multiply(cos_phi, sin_delta_cos)
        x = tf.multiply(x1, x2)
        # y = np.sin(theta + delta * sin_alphas_shifted)
        y1 = tf.multiply(sin_theta, cos_delta_sin)
        y2 = tf.multiply(cos_theta, sin_delta_sin)
        y = y1 + y2

        ix = e_x0 + tf.multiply(eyeball_radius, x)
        iy = e_y0 + tf.multiply(eyeball_radius, y)
        dx = tf.subtract(ix, iris_landmarks[:, 0])
        dy = tf.subtract(iy, iris_landmarks[:, 1])
        out = tf.reduce_mean(tf.square(dx) + tf.square(dy))

        # In addition, match estimated and actual iris centre
        iris_dx = e_x0 + tf.subtract(tf.multiply(tf.multiply(eyeball_radius, -cos_theta), sin_phi), i_x0)
        iris_dy = e_y0 + tf.subtract(tf.multiply(eyeball_radius, sin_theta), i_y0)
        out += tf.square(iris_dx) + tf.square(iris_dy)

        # sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
        # cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase
        dsin_alphas_shifted_dphase = tf.multiply(-sin_alphas, sin_phase) + tf.multiply(cos_alphas, cos_phase)
        dcos_alphas_shifted_dphase = tf.subtract(tf.multiply(-cos_alphas, sin_phase), tf.multiply(sin_alphas, cos_phase))

        # sin_delta_sin = np.sin(delta * sin_alphas_shifted)
        # sin_delta_cos = np.sin(delta * cos_alphas_shifted)
        # cos_delta_sin = np.cos(delta * sin_alphas_shifted)
        # cos_delta_cos = np.cos(delta * cos_alphas_shifted)
        dsin_delta_sin_ddelta = tf.multiply(cos_delta_sin, sin_alphas_shifted)
        dsin_delta_cos_ddelta = tf.multiply(cos_delta_cos, cos_alphas_shifted)
        dcos_delta_sin_ddelta = tf.multiply(-sin_delta_sin, sin_alphas_shifted)
        dcos_delta_cos_ddelta = tf.multiply(-sin_delta_cos, cos_alphas_shifted)
        dsin_delta_sin_dphase = tf.multiply(tf.multiply(cos_delta_sin, delta), dsin_alphas_shifted_dphase)
        dsin_delta_cos_dphase = tf.multiply(tf.multiply(cos_delta_cos, delta), dcos_alphas_shifted_dphase)
        dcos_delta_sin_dphase = tf.multiply(tf.multiply(-sin_delta_sin, delta), dsin_alphas_shifted_dphase)
        dcos_delta_cos_dphase = tf.multiply(tf.multiply(-sin_delta_cos, delta), dcos_alphas_shifted_dphase)

        # x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
        # x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
        dx1_dtheta = tf.multiply(sin_theta, cos_delta_sin) + tf.multiply(cos_theta, sin_delta_sin)
        dx2_dtheta = 0.0
        dx1_dphi = 0.0
        dx2_dphi = tf.subtract(tf.multiply(cos_phi, cos_delta_cos), tf.multiply(sin_phi, sin_delta_cos))
        dx1_ddelta = tf.multiply(-cos_theta, dcos_delta_sin_ddelta) + tf.multiply(sin_theta, dsin_delta_sin_ddelta)
        dx2_ddelta = tf.multiply(sin_phi, dcos_delta_cos_ddelta) + tf.multiply(cos_phi, dsin_delta_cos_ddelta)
        dx1_dphase = tf.multiply(-cos_theta, dcos_delta_sin_dphase) + tf.multiply(sin_theta, dsin_delta_sin_dphase)
        dx2_dphase = tf.multiply(sin_phi, dcos_delta_cos_dphase) + tf.multiply(cos_phi, dsin_delta_cos_dphase)

        # y1 = sin_theta * cos_delta_sin
        # y2 = cos_theta * sin_delta_sin
        dy1_dtheta = tf.multiply(cos_theta, cos_delta_sin)
        dy2_dtheta = tf.multiply(-sin_theta, sin_delta_sin)
        dy1_dphi = 0.0
        dy2_dphi = 0.0
        dy1_ddelta = tf.multiply(sin_theta, dcos_delta_sin_ddelta)
        dy2_ddelta = tf.multiply(cos_theta, dsin_delta_sin_ddelta)
        dy1_dphase = tf.multiply(sin_theta, dcos_delta_sin_dphase)
        dy2_dphase = tf.multiply(cos_theta, dsin_delta_sin_dphase)

        # x = x1 * x2
        # y = y1 + y2
        dx_dtheta = tf.multiply(dx1_dtheta, x2) + tf.multiply(x1, dx2_dtheta)
        dx_dphi = tf.multiply(dx1_dphi, x2) + tf.multiply(x1, dx2_dphi)
        dx_ddelta = tf.multiply(dx1_ddelta, x2) + tf.multiply(x1, dx2_ddelta)
        dx_dphase = tf.multiply(dx1_dphase, x2) + tf.multiply(x1, dx2_dphase)
        dy_dtheta = dy1_dtheta + dy2_dtheta
        dy_dphi = dy1_dphi + dy2_dphi
        dy_ddelta = dy1_ddelta + dy2_ddelta
        dy_dphase = dy1_dphase + dy2_dphase

        # ix = w_2 + eyeball_radius * x
        # iy = h_2 + eyeball_radius * y
        dix_dtheta = tf.multiply(eyeball_radius, dx_dtheta)
        dix_dphi = tf.multiply(eyeball_radius, dx_dphi)
        dix_ddelta = tf.multiply(eyeball_radius, dx_ddelta)
        dix_dphase = tf.multiply(eyeball_radius, dx_dphase)
        diy_dtheta = tf.multiply(eyeball_radius, dy_dtheta)
        diy_dphi = tf.multiply(eyeball_radius, dy_dphi)
        diy_ddelta = tf.multiply(eyeball_radius, dy_ddelta)
        diy_dphase = tf.multiply(eyeball_radius, dy_dphase)

        # dx = ix - iris_landmarks[:, 0]
        # dy = iy - iris_landmarks[:, 1]
        ddx_dtheta = dix_dtheta
        ddx_dphi = dix_dphi
        ddx_ddelta = dix_ddelta
        ddx_dphase = dix_dphase
        ddy_dtheta = diy_dtheta
        ddy_dphi = diy_dphi
        ddy_ddelta = diy_ddelta
        ddy_dphase = diy_dphase

        # out = dx ** 2 + dy ** 2

        dim = tf.constant([2])
        dim = tf.to_float(dim)

        dout_dtheta, _ = tf.nn.moments(tf.multiply(dim, (tf.multiply(dx, ddx_dtheta) + tf.multiply(dy, ddy_dtheta))), 0)
        dout_dphi, _ = tf.nn.moments(tf.multiply(dim, (tf.multiply(dx, ddx_dphi) + tf.multiply(dy, ddy_dphi))), 0)
        dout_ddelta, _ = tf.nn.moments(tf.multiply(dim, (tf.multiply(dx, ddx_ddelta) + tf.multiply(dy, ddy_ddelta))), 0)
        dout_dphase, _ = tf.nn.moments(tf.multiply(dim, (tf.multiply(dx, ddx_dphase) + tf.multiply(dy, ddy_dphase))), 0)

        # iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
        # iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
        # out += iris_dx ** 2 + iris_dy ** 2
        dout_dtheta += tf.multiply(dim, tf.multiply(eyeball_radius, (tf.multiply(sin_theta, tf.multiply(sin_phi, iris_dx)) + tf.multiply(cos_theta, iris_dy))))
        dout_dphi += tf.multiply(dim, tf.multiply(eyeball_radius, (tf.multiply(-cos_theta, tf.multiply(cos_phi, iris_dx)))))

        '''
        print(dout_dtheta)
        print(dout_dphi)
        print(dout_ddelta)
        print(dout_dphase)

        kak = [dout_dtheta, dout_dphi, dout_ddelta, dout_dphase]
        print(kak)'''


        return out, [dout_dtheta, dout_dphi, dout_ddelta, dout_dphase]

    phase = 0.02

    delta = tf.constant([0.1 * np.pi])
    phase = tf.constant([0.02])

    
    '''
    result = scipy.optimize.minimize(gaze_fit_loss_func_tf, x0=[theta, phi, delta, phase],
                                     bounds=(
                                         (-0.4*np.pi, 0.4*np.pi),
                                         (-0.4*np.pi, 0.4*np.pi),
                                         (0.01*np.pi, 0.5*np.pi),
                                         (-np.pi, np.pi),
                                     ),
                                     jac=True,
                                     tol=1e-6,
                                     method='TNC',
                                     options={
                                         # 'disp': True,
                                         'gtol': 1e-6,
                                         'maxiter': 100,
                                    })
    '''
    #sloss = gaze_fit_loss_func_tf(x0=[theta, phi, delta, phase])
    #result = tfp.optimizer.bfgs_minimize(gaze_fit_loss_func_tf, [theta, phi, delta, phase])
    result = newton.newton_dou(gaze_fit_loss_func_tf, [theta, phi, delta, phase])

    #if result.success:
        #theta, phi, delta, phase = result.x

    #result
    '''
    print("6666666666666666666666666666666666")
    a = tf.constant([1,2,3])
    b = tf.as_string(theta)
    print(a)
    print(theta)
    print(b)
    with tf.Session() as s:
        qqq=theta.eval()
    #qqq = tf.Session().run(a.eval())
    print(qqq)

    result = scipy.optimize.minimize(gaze_fit_loss_func_tf, x0 = [theta, phi, delta, phase], method='TNC', jac = True, 
        bounds = ((-0.4*np.pi, 0.4*np.pi), (-0.4*np.pi, 0.4*np.pi), (0.01*np.pi, 0.5*np.pi), (-np.pi, np.pi),), tol = 1e-6, options = {'maxiter': 100, 'gtol': 1e-6})

    if result.success:
        theta, phi, delta, phase = result.x'''

    #return np.array([-theta, phi])
    return tf.constant([-theta, phi])
