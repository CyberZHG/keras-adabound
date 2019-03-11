# Keras AdaBound

[![Travis](https://travis-ci.org/CyberZHG/keras-adabound.svg)](https://travis-ci.org/CyberZHG/keras-adabound)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-adabound/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-adabound)

[AdaBound](https://github.com/Luolc/AdaBound) optimizer in Keras.

## Install

```bash
pip install keras-adabound
```

## Usage

### Use the optimizer

```python
from keras_adabound import AdaBound

model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss=model_loss)
```

### Load with custom objects

```python
from keras_adabound import AdaBound

model = keras.models.load_model(model_path, custom_objects={'AdaBound': AdaBound})
```

### About weight decay

The optimizer does not have an argument named `weight_decay` (as in the official repo) since it can be done by adding L2 regularizers to weights:

```python
import keras

regularizer = keras.regularizers.l2(WEIGHT_DECAY / 2)
for layer in model.layers:
    for attr in ['kernel_regularizer', 'bias_regularizer']:
        if hasattr(layer, attr) and layer.trainable:
            setattr(layer, attr, regularizer)
```
