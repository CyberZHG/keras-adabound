from .backend import keras
from .backend import backend as K


class AdaBound(keras.optimizers.Optimizer):
    """AdamBound optimizer.

    # Arguments
        lr: float >= 0. Learning rate.
        final_lr: float >= 0. Final (SGD) learning rate.
        base_lr: float >= 0. Used for loading the optimizer. Do not set the argument manually.
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

    def __init__(self, lr=0.001, final_lr=0.1, base_lr=None,
                 beta_1=0.9, beta_2=0.999, gamma=0.001,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False, **kwargs):
        super(AdaBound, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.final_lr = K.variable(final_lr, name='final_lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.gamma = K.variable(gamma, name='gamma')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        if base_lr is None:
            self.base_lr = lr
        else:
            self.base_lr = base_lr
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))
        final_lr = self.final_lr * lr / self.base_lr
        lower_bound = final_lr * (1.0 - 1.0 / (self.gamma * t + 1.0))
        upper_bound = final_lr * (1.0 + 1.0 / (self.gamma * t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            if self.initial_weight_decay > 0.:
                # Note that the decayed weights are added to the momentums.
                # The mechanism is the same as the official repo.
                g += self.weight_decay * p

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                step = lr_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                step = lr_t / (K.sqrt(v_t) + self.epsilon)
            p_t = p - K.minimum(K.maximum(step, lower_bound), upper_bound) * m_t
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'final_lr': float(K.get_value(self.final_lr)),
                  'base_lr': self.base_lr,
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'gamma': float(K.get_value(self.gamma)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdaBound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
