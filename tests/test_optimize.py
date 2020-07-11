import os
import tempfile
from unittest import TestCase
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_adabound import AdaBound


K = keras.backend


class TestOptimizers(TestCase):

    @staticmethod
    def reset_seed(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def gen_keras_linear(w, b, amsgrad=False):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(input_shape=(3,), units=5, weights=[w, b]))
        model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1, amsgrad=amsgrad, weight_decay=1e-3), loss='mse')
        return model

    @staticmethod
    def gen_random_weights():
        return np.random.standard_normal((3, 5)), np.random.standard_normal((5,))

    def test_with_scheduler(self):
        w, b = self.gen_random_weights()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            input_shape=(3,),
            units=5,
            weights=[w, b]),
        )
        decay = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96)
        decay = tf.keras.optimizers.schedules.serialize(decay)
        model.compile(optimizer=AdaBound(learning_rate=decay,
                                         final_lr=0.1,
                                         decay=0.5,
                                         weight_decay=decay), loss='mse')
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
        with tempfile.TemporaryDirectory() as temp_path:
            model_path = os.path.join(temp_path, 'keras_adabound.h5')
            model.save(model_path)
            model = keras.models.load_model(model_path, custom_objects={'AdaBound': AdaBound})
        self.assertGreater(1e-3, float(K.get_value(model.optimizer.lr)))
        self.assertEqual(1e-3, model.optimizer.base_lr)

    def _embedding_data(self):
        while True:
            x = np.random.randint(0, 10, (3, 7))
            y = np.zeros(3)
            for i in range(3):
                if 5 in x[i]:
                    y[i] = 1
            yield x, y

    def test_with_embedding(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(
            input_dim=10,
            output_dim=5,
            mask_zero=True,
            input_shape=(7,)),
        )
        model.add(keras.layers.LSTM(units=5))
        model.add(keras.layers.Dense(units=2, activation='softmax'))
        model.compile(optimizer=AdaBound(), loss='sparse_categorical_crossentropy')
        model.fit(self._embedding_data(),
                  steps_per_epoch=1000,
                  validation_data=self._embedding_data(),
                  validation_steps=10,
                  epochs=3)

    def test_with_embedding_amsgrad(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(
            input_dim=10,
            mask_zero=True,
            output_dim=5,
            input_shape=(7,)),
        )
        model.add(keras.layers.LSTM(units=5))
        model.add(keras.layers.Dense(units=2, activation='softmax'))
        model.compile(optimizer=AdaBound(amsgrad=True,
                                         weight_decay=1e-3), loss='sparse_categorical_crossentropy')
        model.fit(self._embedding_data(),
                  steps_per_epoch=1000,
                  validation_data=self._embedding_data(),
                  validation_steps=10,
                  epochs=2)
        with tempfile.TemporaryDirectory() as temp_path:
            model_path = os.path.join(temp_path, 'keras_adabound.h5')
            model.save(model_path)
            model = keras.models.load_model(model_path, custom_objects={'AdaBound': AdaBound})
        model.fit(self._embedding_data(),
                  steps_per_epoch=1000,
                  validation_data=self._embedding_data(),
                  validation_steps=10,
                  epochs=1)
