"""Concurrent testing during training."""
import collections
import platform
import threading
import time
import traceback

import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class LiveTester(object):
    """Manage concurrent testing on test data source."""

    def __init__(self, model, data_source, use_batch_statistics=True):
        """Initialize tester with reference to model and data sources."""
        self.model = model
        self.data = data_source
        self.time = self.model.time
        self.summary = self.model.summary
        self._tensorflow_session = model._tensorflow_session

        self._is_testing = False
        self._condition = threading.Condition()

        self._use_batch_statistics = use_batch_statistics

    def stop(self):
        logger.info('LiveTester::stop is being called.')
        self._is_testing = False

    def __del__(self):
        """Handle deletion of instance by closing thread."""
        if not hasattr(self, '_coordinator'):
            return
        self._coordinator.request_stop()
        with self._condition:
            self._is_testing = True  # Break wait if waiting
            self._condition.notify_all()
        self._coordinator.join([self._thread], stop_grace_period_secs=1)

    def _true_if_testing(self):
        return self._is_testing

    def trigger_test_if_not_testing(self, current_step):
        """If not currently testing, run test."""
        if not self._is_testing:
            with self._condition:
                self._is_testing = True
                self._testing_at_step = current_step
                self._condition.notify_all()

    def test_job(self):
        """Evaluate requested metric over entire test set."""
        while not self._coordinator.should_stop():
            with self._condition:
                self._condition.wait_for(self._true_if_testing)
                if self._coordinator.should_stop():
                    break
                should_stop = False
                try:
                    should_stop = self.do_full_test()
                except:
                    traceback.print_exc()
                self._is_testing = False
                if should_stop is True:
                    break
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def do_full_test(self, sleep_between_batches=0.2):
        # Copy current weights over
        self.copy_model_weights()

        # Reset data sources
        for data_source_name, data_source in self.data.items():
            data_source.reset()
            num_batches = int(data_source.num_entries / data_source.batch_size)

        # Decide what to evaluate
        fetches = self._tensors_to_evaluate
        outputs = dict([(name, list()) for name in fetches.keys()])

        # Select random index to produce (image) summaries at
        summary_index = np.random.randint(num_batches)

        self.time.start('full test')
        for i in range(num_batches):
            if self._is_testing is not True:
                logger.debug('Testing flag found to be `False` at iter. %d' % i)
                break
            logger.debug('Testing on %03d/%03d batches.' % (i + 1, num_batches))
            if i == summary_index:
                fetches['summaries'] = self.summary.get_ops(mode='test')
            try:
                output = self._tensorflow_session.run(
                    fetches=fetches,
                    feed_dict={
                        self.model.is_training: False,
                        self.model.use_batch_statistics: self._use_batch_statistics,
                    },
                )
            except (tf.errors.CancelledError, RuntimeError):
                return True
            time.sleep(sleep_between_batches)  # Brief pause to prioritise training
            if 'summaries' in output:  # Write summaries on first batch
                self.summary.write_summaries(output['summaries'], self._testing_at_step)
                del fetches['summaries']
                del output['summaries']
            for name, value in output.items():  # Gather results from this batch
                outputs[name].append(output[name])

            metrics = {}
            loss_terms = {}

            while True:
                out = next(output)

                heatmaps_amax = np.amax(out['heatmaps'][j, :].reshape(-1, 18), axis=0)
                can_use_eye = np.all(heatmaps_amax > 0.7)
                can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
                can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

                eye_landmarks = out['landmarks'][j, :]
                eye_radius = out['radius'][j][0]
                if eye_side == 'left':
                    eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                    eye_image = np.fliplr(eye_image)

                face_index = int(eye_index / 2)
                eh, ew, _ = eye_image_raw.shape
                v0 = face_index * 2 * eh
                v1 = v0 + eh
                v2 = v1 + eh
                u0 = 0 if eye_side == 'left' else ew
                u1 = u0 + ew
                bgr[v0:v1, u0:u1] = eye_image_raw
                bgr[v1:v2, u0:u1] = eye_image_annotated

                # Transform predictions
                eye_landmarks = np.concatenate([eye_landmarks,
                                                [[eye_landmarks[-1, 0] + eye_radius,
                                                  eye_landmarks[-1, 1]]]])
                eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                   'constant', constant_values=1.0))
                eye_landmarks = (eye_landmarks *
                                 eye['inv_landmarks_transform_mat'].T)[:, :2]
                eye_landmarks = np.asarray(eye_landmarks)
                eyelid_landmarks = eye_landmarks[0:8, :]
                iris_landmarks = eye_landmarks[8:16, :]
                iris_centre = eye_landmarks[16, :]
                eyeball_centre = eye_landmarks[17, :]
                eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                                eye_landmarks[17, :])


                x = estimate_gaze_from_landmarks(iris_landmarks, iris_centre, eyeball_centre, eyeball_radius,
                                         initial_gaze=None)

                data_source = next(iter(Testdata.values()))
                input_tensors = data_source.output_tensors
                y1 = input_tensors['gazemaps'] if 'gazemaps' in input_tensors else None
                y2 = input_tensors['gaze'] if 'gaze' in input_tensors else None

                if y1 is not None:
                    # Cross-entropy loss
                    metrics['gazemaps_ce'] = -tf.reduce_mean(tf.reduce_sum(
                        y1 * tf.log(tf.clip_by_value(gmap, 1e-10, 1.0)),  # avoid NaN
                        axis=[1, 2, 3]))

                if y2 is not None:
                    metrics['gaze_mse'] = tf.reduce_mean(tf.squared_difference(x, y2))
                    metrics['gaze_ang'] = tensorflow_angular_error_from_pitchyaw(y2, x)

                if y1 is not None and y2 is not None:
                    loss_terms['combined_loss'] = 1e-5*metrics['gazemaps_ce'] + metrics['gaze_mse']

            for combined_loss in loss_terms.items():  
                outputs[loss].append(loss_terms[combined_loss])

            for gazemaps_ce, gaze_mse in metrics.items():  
                outputs[gazemapce].append(metrics[gazemaps_ce])
                outputs[gazemse].append(metrics[gaze_mse])


        self.time.end('full test')

        # If incomplete, skip this round of tests (most likely shutting down)
        if len(list(outputs.values())[0]) != num_batches:
            return True

        # Calculate mean values
        for name, values in outputs.items():
            outputs[name] = np.mean(values)

        # TODO: Log metric as summary
        to_print = '[Test at step %06d] ' % self._testing_at_step
        to_print += ', '.join([
            '%s = %f' % (name, value, loss, gazemapce, gazemse) for name, value, loss, gazemapce, gazemse in outputs.items()
        ])
        logger.info(to_print)

        # Store mean metrics/losses (and other summaries)
        feed_dict = dict([(self._placeholders[name], value, loss, gazemapce, gazemse)
                         for name, value, loss, gazemapce, gazemse in outputs.items()])
        feed_dict[self.model.is_training] = False
        feed_dict[self.model.use_batch_statistics] = True
        try:
            summaries = self._tensorflow_session.run(
                fetches=self.summary.get_ops(mode='full_test'),
                feed_dict=feed_dict,
            )
        except (tf.errors.CancelledError, RuntimeError):
            return True
        self.summary.write_summaries(summaries, self._testing_at_step)

        return False

    def do_final_full_test(self, current_step):
        logger.info('Stopping the live testing threads.')

        # Stop thread(s)
        self._is_testing = False
        self._coordinator.request_stop()
        with self._condition:
            self._is_testing = True  # Break wait if waiting
            self._condition.notify_all()
        self._coordinator.join([self._thread], stop_grace_period_secs=1)

        # Start final full test
        logger.info('Running final full test')
        self.copy_model_weights()
        self._is_testing = True
        self._testing_at_step = current_step
        self.do_full_test(sleep_between_batches=0)

    def _post_model_build(self):
        """Prepare combined operation to copy model parameters over from CPU/GPU to CPU."""
        with tf.variable_scope('copy2test'):
            all_variables = tf.global_variables()
            train_vars = dict([(v.name, v) for v in all_variables
                               if not v.name.startswith('test/')])
            test_vars = dict([(v.name, v) for v in all_variables
                              if v.name.startswith('test/')])
            self._copy_variables_to_test_model_op = tf.tuple([
                test_vars['test/' + k].assign(train_vars[k]) for k in train_vars.keys()
                if 'test/' + k in test_vars
            ])

        # Begin testing thread
        self._coordinator = tf.train.Coordinator()
        self._thread = threading.Thread(target=self.test_job,
                                        name='%s_tester' % self.model.identifier)
        self._thread.daemon = True
        self._thread.start()

        # Pick tensors we need to evaluate
        all_tensors = dict(self.model.loss_terms['test'], **self.model.metrics['test'])
        self._tensors_to_evaluate = dict([(n, t) for n, t in all_tensors.items()])
        loss_terms_to_evaluate = dict([(n, t) for n, t in self.model.loss_terms['test'].items()
                                       if t in self._tensors_to_evaluate.values()])
        metrics_to_evaluate = dict([(n, t) for n, t in self.model.metrics['test'].items()
                                    if t in self._tensors_to_evaluate.values()])

        # Placeholders for writing summaries at end of test run
        self._placeholders = {}
        for type_, tensors in (('loss', loss_terms_to_evaluate),
                               ('metric', metrics_to_evaluate)):
            for name in tensors.keys():
                name = '%s/test/%s' % (type_, name)
                placeholder = tf.placeholder(dtype=np.float32, name=name + '_placeholder')
                self.summary.scalar(name, placeholder)
                self._placeholders[name.split('/')[-1]] = placeholder

    def copy_model_weights(self):
        """Copy weights from main model used for training.

        This operation should stop-the-world, that is, training should not occur.
        """
        assert self._copy_variables_to_test_model_op is not None
        self._tensorflow_session.run(self._copy_variables_to_test_model_op)
        logger.debug('Copied over trainable model parameters for testing.')

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



    def estimate_gaze_from_landmarks(iris_landmarks, iris_centre, eyeball_centre, eyeball_radius,
                                 initial_gaze=None):
        e_x0, e_y0 = eyeball_centre
        i_x0, i_y0 = iris_centre

        if initial_gaze is not None:
            theta, phi = initial_gaze
            # theta = -theta
        else:
            theta = np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))

        delta = 0.1 * np.pi
        if iris_landmarks[0, 0] < iris_landmarks[4, 0]:  # flipped
            alphas = np.flip(np.arange(0.0, 2.0 * np.pi, step=np.pi/4.0), axis=0)
        else:
            alphas = np.arange(-np.pi, np.pi, step=np.pi/4.0) + np.pi/4.0
        sin_alphas = np.sin(alphas)
        cos_alphas = np.cos(alphas)

        def gaze_fit_loss_func(inputs):
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

        phase = 0.02
        result = scipy.optimize.minimize(gaze_fit_loss_func, x0=[theta, phi, delta, phase],
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
        if result.success:
            theta, phi, delta, phase = result.x

        return np.array([-theta, phi])

