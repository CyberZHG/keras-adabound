import os
import tempfile
import random
from unittest import TestCase
import keras
import keras.backend as K
import torch
import numpy as np
import tensorflow as tf
from adabound import AdaBound as OfficialAdaBound
from keras_adabound import AdaBound


class TestOptimizers(TestCase):

    @staticmethod
    def reset_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def gen_torch_linear(w, b):
        linear = torch.nn.Linear(3, 5)
        linear.weight = torch.nn.Parameter(torch.Tensor(w.transpose().tolist()))
        linear.bias = torch.nn.Parameter(torch.Tensor(b.tolist()))
        return linear

    @staticmethod
    def gen_keras_linear(w, b, amsgrad=False):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(input_shape=(3,), units=5, weights=[w, b]))
        model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1, amsgrad=amsgrad), loss='mse')
        return model

    @staticmethod
    def gen_random_weights():
        return np.random.standard_normal((3, 5)), np.random.standard_normal((5,))

    def test_same(self):
        self.reset_seed(0xcafe)
        w, b = self.gen_random_weights()
        torch_linear = self.gen_torch_linear(w, b)
        keras_linear = self.gen_keras_linear(w, b)
        model_path = os.path.join(tempfile.gettempdir(), 'keras_adabound.h5')
        keras_linear.save(model_path)
        keras_linear = keras.models.load_model(model_path, custom_objects={'AdaBound': AdaBound})
        w, b = self.gen_random_weights()
        criterion = torch.nn.MSELoss()
        optimizer = OfficialAdaBound(torch_linear.parameters(), lr=1e-3, final_lr=0.1, eps=K.epsilon())
        for i in range(300):
            x = np.random.standard_normal((1, 3))
            y = np.dot(x, w) + b
            optimizer.zero_grad()
            y_hat = torch_linear(torch.Tensor(x.tolist()))
            loss = criterion(y_hat, torch.Tensor(y.tolist()))
            torch_loss = loss.tolist()
            loss.backward()
            optimizer.step()
            keras_loss = keras_linear.train_on_batch(x, y).tolist()
            # print(i, torch_loss, keras_loss)
        self.assertTrue(abs(torch_loss - keras_loss) < 1e-2)
        self.assertTrue(np.allclose(
            torch_linear.weight.detach().numpy().transpose(),
            keras_linear.get_weights()[0],
            atol=1e-2,
        ))
        self.assertTrue(np.allclose(
            torch_linear.bias.detach().numpy(),
            keras_linear.get_weights()[1],
            atol=1e-2,
        ))

    def test_same_amsgrad(self):
        self.reset_seed(0xcafe)
        w, b = self.gen_random_weights()
        torch_linear = self.gen_torch_linear(w, b)
        keras_linear = self.gen_keras_linear(w, b, amsgrad=True)
        w, b = self.gen_random_weights()
        criterion = torch.nn.MSELoss()
        optimizer = OfficialAdaBound(
            torch_linear.parameters(),
            lr=1e-3,
            final_lr=0.1,
            eps=K.epsilon(),
            amsbound=True,
        )
        for i in range(300):
            x = np.random.standard_normal((1, 3))
            y = np.dot(x, w) + b
            optimizer.zero_grad()
            y_hat = torch_linear(torch.Tensor(x.tolist()))
            loss = criterion(y_hat, torch.Tensor(y.tolist()))
            torch_loss = loss.tolist()
            loss.backward()
            optimizer.step()
            keras_loss = keras_linear.train_on_batch(x, y).tolist()
            # print(i, torch_loss, keras_loss)
        self.assertTrue(abs(torch_loss - keras_loss) < 1e-2)
        self.assertTrue(np.allclose(
            torch_linear.weight.detach().numpy().transpose(),
            keras_linear.get_weights()[0],
            atol=1e-2,
        ))
        self.assertTrue(np.allclose(
            torch_linear.bias.detach().numpy(),
            keras_linear.get_weights()[1],
            atol=1e-2,
        ))
