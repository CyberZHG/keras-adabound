import os
import tempfile
from unittest import TestCase
import numpy as np
import tensorflow as tf
from keras_adabound.backend import keras
from keras_adabound.backend import backend as K
from keras_adabound import AdaBound


class TestOptimizers(TestCase):

    @staticmethod
    def reset_seed(seed):
        np.random.seed(seed)
        tf.set_random_seed(seed)

    @staticmethod
    def gen_keras_linear(w, b, amsgrad=False):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(input_shape=(3,), units=5, weights=[w, b]))
        model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1, amsgrad=amsgrad, weight_decay=1e-3), loss='mse')
        return model

    @staticmethod
    def gen_random_weights():
        return np.random.standard_normal((3, 5)), np.random.standard_normal((5,))

    def test_with_constraint(self):
        w, b = self.gen_random_weights()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            input_shape=(3,),
            units=5,
            kernel_constraint=keras.constraints.max_norm(1.0),
            weights=[w, b]),
        )
        model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1, decay=0.5), loss='mse')
        x = np.random.standard_normal((1, 3))
        y = np.dot(x, w) + b
        model.train_on_batch(x, y)

    def test_with_plateau(self):
        self.reset_seed(0xcafe)
        w, b = self.gen_random_weights()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            input_shape=(3,),
            units=5,
            weights=[w, b]),
        )
        model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='mse')
        x = np.random.standard_normal((10000, 3))
        y = np.dot(x, w) + b
        model.fit(x, y, epochs=100, callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='loss')], verbose=False)
        model_path = os.path.join(tempfile.gettempdir(), 'keras_adabound_plateau.h5')
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'AdaBound': AdaBound})
        self.assertGreater(1e-3, float(K.get_value(model.optimizer.lr)))
        self.assertEqual(1e-3, model.optimizer.base_lr)
