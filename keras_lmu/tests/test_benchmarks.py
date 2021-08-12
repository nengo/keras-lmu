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
@pytest.mark.parametrize("mode", ["rnn", "fft", "raw"])
def test_performance(mode, capsys):
    dims = 32
    seq_len = 1024
    batch_size = 32
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
    model.predict(x_train, batch_size=batch_size, callbacks=[steptimes])

    with capsys.disabled():
        print(f"train step time: {np.min(steptimes.train_times):0.4f}")
        print(f"inference step time: {np.min(steptimes.inference_times):0.4f}")
