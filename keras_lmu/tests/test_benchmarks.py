# pylint: disable=missing-docstring

import timeit

import keras
import numpy as np
import pytest
import tensorflow as tf

from keras_lmu import layers
from keras_lmu.tests import tf_gpu_installed


class SteptimeLogger(keras.callbacks.Callback):
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
    [("rnn", 0.01, 0.1), ("fft", 0.01, 0.1), ("raw", 0.01, 0.1)],
)
def test_performance(mode, min_time, max_time):
    # performance is based on
    # CPU: AMD Ryzen 9 5950X
    # GPU: Nvidia RTX 3060
    # TensorFlow version: 2.10.0

    dims = 32
    seq_len = 512
    batch_size = 16
    odims = 2

    kwargs = {"memory_d": dims, "order": 256, "theta": 784, "hidden_cell": None}
    if mode == "rnn":
        lmu_layer = keras.layers.RNN(
            layers.LMUCell(**kwargs),
            return_sequences=False,
        )
    else:
        assert mode in ["fft", "raw"]
        lmu_layer = layers.LMUFeedforward(
            return_sequences=False, conv_mode=mode, **kwargs
        )

    inputs = keras.layers.Input((seq_len, dims), batch_size=batch_size)
    lmu = lmu_layer(inputs)
    outputs = keras.layers.Dense(odims)(lmu)

    model = keras.Model(inputs=inputs, outputs=outputs)

    n_train = 20 * batch_size
    x_train = tf.random.uniform((n_train, seq_len, dims), minval=-1, maxval=1, seed=0)
    y_train = tf.random.uniform((n_train, odims), minval=-1, maxval=1, seed=1)
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.RMSprop(),
    )

    steptimes = SteptimeLogger()
    model.fit(x_train, y_train, epochs=1, batch_size=batch_size, callbacks=[steptimes])

    step_time = np.min(steptimes.train_times)
    assert step_time >= min_time
    assert step_time <= max_time
