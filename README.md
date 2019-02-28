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
