# pylint: disable=missing-docstring

import keras
import tensorflow as tf
from packaging import version


def pytest_configure(config):
    tf.debugging.disable_traceback_filtering()
    tf.keras.utils.set_random_seed(0)
    if version.parse(tf.__version__) >= version.parse("2.16.0"):
        keras.config.disable_traceback_filtering()
