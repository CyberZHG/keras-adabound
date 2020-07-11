from typing import Union, Callable, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typeguard import typechecked


K = keras.backend
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]


class AdaBound(keras.optimizers.Optimizer):
    """AdamBound optimizer.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        base_lr: float >= 0. Used for loading the optimizer. Do not set the argument manually.
        final_lr: float >= 0. Final (SGD) learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float, 0 < gamma < 1. Convergence speed of the bound functions.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm.

    # References
        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate]
          (https://openreview.net/forum?id=Bkg3g2R9FX)
    """

    @typechecked
    def __init__(
            self,
            learning_rate: Union[FloatTensorLike, Callable, Dict] = 0.001,
            base_lr: Optional[FloatTensorLike] = None,
            final_lr: FloatTensorLike = 0.1,
            beta_1: FloatTensorLike = 0.9,
            beta_2: FloatTensorLike = 0.999,
            gamma: FloatTensorLike = 0.001,
            epsilon: FloatTensorLike = 1e-8,
            weight_decay: Union[FloatTensorLike, Callable, Dict] = 0.0,
            amsgrad: bool = False,
            name: str = "AdaBound",
            **kwargs
    ):
        super(AdaBound, self).__init__(name=name, **kwargs)

        if isinstance(learning_rate, Dict):
            learning_rate = tf.keras.optimizers.schedules.deserialize(learning_rate)

        if isinstance(weight_decay, Dict):
            weight_decay = tf.keras.optimizers.schedules.deserialize(weight_decay)

        if base_lr is None:
            if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                base_lr = learning_rate(0)
            else:
                base_lr = learning_rate

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("base_lr", base_lr)
        self._set_hyper("final_lr", final_lr)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("gamma", gamma)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self._has_weight_decay = weight_decay != 0.0

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        base_lr = self._get_hyper("base_lr", var_dtype)
        final_lr = self._get_hyper("final_lr", var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        gamma = self._get_hyper("gamma", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._has_weight_decay:
            grad += wd_t * var

        m_t = m.assign(beta_1_t * m + (1.0 - beta_1_t) * grad, use_locking=self._use_locking)
        v_t = v.assign(beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad), use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            denom = tf.sqrt(vhat_t) + epsilon_t
        else:
            vhat_t = None
            denom = tf.sqrt(v_t) + epsilon_t

        final_lr = final_lr * lr_t / base_lr
        lower_bound = final_lr * (1.0 - 1.0 / (gamma * local_step + 1.0))
        upper_bound = final_lr * (1.0 + 1.0 / (gamma * local_step))
        lr_t = lr_t * (tf.sqrt(1.0 - beta_2_power) / (1.0 - beta_1_power))
        lr_t = tf.clip_by_value(lr_t / denom, lower_bound, upper_bound)
        var_update = var.assign_sub(lr_t * m_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        base_lr = self._get_hyper("base_lr", var_dtype)
        final_lr = self._get_hyper("final_lr", var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        gamma = self._get_hyper("gamma", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._has_weight_decay:
            grad = grad + wd_t * tf.squeeze(tf.gather(tf.expand_dims(var, axis=0), indices, axis=1), axis=0)

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = m.assign(m * beta_1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = v.assign(v * beta_2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            denom = tf.sqrt(vhat_t) + epsilon_t
        else:
            vhat_t = None
            denom = tf.sqrt(v_t) + epsilon_t

        final_lr = final_lr * lr_t / base_lr
        lower_bound = final_lr * (1.0 - 1.0 / (gamma * local_step + 1.0))
        upper_bound = final_lr * (1.0 + 1.0 / (gamma * local_step))
        lr_t = lr_t * (tf.sqrt(1.0 - beta_2_power) / (1.0 - beta_1_power))
        lr_t = tf.clip_by_value(lr_t / denom, lower_bound, upper_bound)
        with tf.control_dependencies([m_t]):
            var_update = self._resource_scatter_add(
                var, indices, tf.gather(-lr_t * m_t, indices)
            )

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "base_lr": self._serialize_hyperparameter("base_lr"),
                "final_lr": self._serialize_hyperparameter("final_lr"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "gamma": self._serialize_hyperparameter("gamma"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad
            }
        )
        return config
