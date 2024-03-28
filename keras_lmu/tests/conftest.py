# pylint: disable=missing-docstring

import keras
import tensorflow as tf
from packaging import version


def pytest_configure(config):
    if version.parse(tf.__version__) >= version.parse("2.7.0"):
        tf.debugging.disable_traceback_filtering()
    if version.parse(tf.__version__) >= version.parse("2.16.0"):
        keras.config.disable_traceback_filtering()
