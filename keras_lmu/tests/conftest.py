# pylint: disable=missing-docstring

import tensorflow as tf
from packaging import version


def pytest_configure(config):
    if version.parse(tf.__version__) >= version.parse("2.7.0"):
        tf.debugging.disable_traceback_filtering()
