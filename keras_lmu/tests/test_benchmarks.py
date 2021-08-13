# pylint: disable=missing-docstring

import timeit

import numpy as np
import pytest
import tensorflow as tf

from keras_lmu import layers
from keras_lmu.tests import tf_gpu_installed


class SteptimeLogger(tf.keras.callbacks.Callback):
    """Callback that records step times."""

    def __init__(self, count_mode="samples", stateful_metrics=None):
        super().__init__()

        self.train_start = None
        self.inference_start = None

        self.train_times = []
        self.inference_times = []

    def on_train_batch_begin(self, batch, logs=None):
        self.train_start = timeit.default_timer()

    def on_train_batch_end(self, batch, logs=None):
        self.train_times.append(timeit.default_timer() - self.train_start)

    def on_predict_batch_begin(self, batch, logs=None):
        self.inference_start = timeit.default_timer()

    def on_predict_batch_end(self, batch, logs=None):
        self.inference_times.append(timeit.default_timer() - self.inference_start)


@pytest.mark.skipif(not tf_gpu_installed, reason="Very slow on CPU")
@pytest.mark.parametrize(
    "mode, min_time, max_time",
    [("rnn", 0.1, 0.2), ("fft", 0.1, 0.2), ("raw", 0.05, 0.15)],
)
def test_performance(mode, min_time, max_time):
    # performance is based on Azure NC6 VM
    # CPU: Intel Xeon E5-2690 v3 @ 2.60Ghz
    # GPU: Nvidia Tesla K80
    # TensorFlow version: 2.6.0

    dims = 32
    seq_len = 512
    batch_size = 16
    odims = 2

    kwargs = dict(memory_d=dims, order=256, theta=784, hidden_cell=None)
    if mode == "rnn":
        lmu_layer = tf.keras.layers.RNN(
            layers.LMUCell(**kwargs),
            return_sequences=False,
        )
    elif mode in ("fft", "raw"):
        lmu_layer = layers.LMUFeedforward(
            return_sequences=False, conv_mode=mode, **kwargs
        )

    inputs = tf.keras.layers.Input((seq_len, dims), batch_size=batch_size)
    lmu = lmu_layer(inputs)
    outputs = tf.keras.layers.Dense(odims)(lmu)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    n_train = 20 * batch_size
    x_train = tf.random.uniform((n_train, seq_len, dims), minval=-1, maxval=1, seed=0)
    y_train = tf.random.uniform((n_train, odims), minval=-1, maxval=1, seed=1)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.RMSprop(),
    )

    steptimes = SteptimeLogger()
    model.fit(x_train, y_train, epochs=1, batch_size=batch_size, callbacks=[steptimes])

    step_time = np.min(steptimes.train_times)
    assert step_time >= min_time
    assert step_time <= max_time
