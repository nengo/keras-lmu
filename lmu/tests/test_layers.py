# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf

from lmu import layers


def test_multivariate_lmu(rng):
    memory_d = 4
    order = 16
    n_steps = 10
    input_d = 32

    input_enc = rng.uniform(0, 1, size=(input_d, memory_d))

    # check that one multivariate LMU is the same as n one-dimensional LMUs (omitting
    # the hidden part)
    inp = tf.keras.Input(shape=(n_steps, input_d))
    multi_lmu = tf.keras.layers.RNN(
        layers.LMUCell(
            memory_d=memory_d,
            order=order,
            theta=n_steps,
            kernel_initializer=tf.initializers.constant(input_enc),
            hidden_cell=tf.keras.layers.SimpleRNNCell(
                units=memory_d * order,
                activation=None,
                kernel_initializer=tf.initializers.constant(np.eye(memory_d * order)),
                recurrent_initializer=tf.initializers.zeros(),
            ),
        ),
        return_sequences=True,
    )(inp)
    lmus = [
        tf.keras.layers.RNN(
            layers.LMUCell(
                memory_d=1,
                order=order,
                theta=n_steps,
                kernel_initializer=tf.initializers.constant(input_enc[:, [i]]),
                hidden_cell=tf.keras.layers.SimpleRNNCell(
                    units=order,
                    activation=None,
                    kernel_initializer=tf.initializers.constant(np.eye(order)),
                    recurrent_initializer=tf.initializers.zeros(),
                ),
            ),
            return_sequences=True,
        )(inp)
        for i in range(memory_d)
    ]

    model = tf.keras.Model(inp, [multi_lmu] + lmus)

    results = model.predict(rng.uniform(0, 1, size=(1, n_steps, input_d)))

    for i in range(memory_d):
        assert np.allclose(
            results[0][..., i * order : (i + 1) * order], results[i + 1], atol=1e-6
        )


def test_layer_vs_cell(rng):
    memory_d = 4
    order = 12
    n_steps = 10
    input_d = 32

    inp = rng.uniform(-1, 1, size=(2, n_steps, input_d))

    lmu_cell = tf.keras.layers.RNN(
        layers.LMUCell(
            memory_d, order, n_steps, tf.keras.layers.SimpleRNNCell(units=64)
        ),
        return_sequences=True,
    )
    cell_out = lmu_cell(inp)

    lmu_layer = layers.LMU(
        memory_d,
        order,
        n_steps,
        tf.keras.layers.SimpleRNNCell(units=64),
        return_sequences=True,
    )
    lmu_layer.build(inp.shape)
    lmu_layer.rnn_layer.set_weights(lmu_cell.get_weights())
    layer_out = lmu_layer(inp)

    for w0, w1 in zip(
        sorted(lmu_cell.weights, key=lambda w: w.shape.as_list()),
        sorted(lmu_layer.weights, key=lambda w: w.shape.as_list()),
    ):
        assert np.allclose(w0.numpy(), w1.numpy())

    assert np.allclose(cell_out, lmu_cell(inp))
    assert np.allclose(cell_out, layer_out)


def test_save_load_weights(rng, tmp_path):
    memory_d = 4
    order = 12
    n_steps = 10
    input_d = 32

    x = rng.uniform(-1, 1, size=(2, n_steps, input_d))

    inp = tf.keras.Input((None, input_d))
    lmu0 = layers.LMU(
        memory_d,
        order,
        n_steps,
        tf.keras.layers.SimpleRNNCell(units=64),
        return_sequences=True,
    )(inp)
    model0 = tf.keras.Model(inp, lmu0)
    out0 = model0(x)

    lmu1 = layers.LMU(
        memory_d,
        order,
        n_steps,
        tf.keras.layers.SimpleRNNCell(units=64),
        return_sequences=True,
    )(inp)
    model1 = tf.keras.Model(inp, lmu1)
    out1 = model1(x)

    assert not np.allclose(out0, out1)

    model0.save_weights(str(tmp_path))
    model1.load_weights(str(tmp_path))

    out2 = model1(x)
    assert np.allclose(out0, out2)


@pytest.mark.parametrize("mode", ("cell", "lmu", "fft"))
def test_save_load_serialization(mode, tmp_path):
    inp = tf.keras.Input((10 if mode == "fft" else None, 32))
    if mode == "cell":
        out = tf.keras.layers.RNN(
            layers.LMUCell(1, 2, 3, tf.keras.layers.SimpleRNNCell(4)),
            return_sequences=True,
        )(inp)
    elif mode == "lmu":
        out = layers.LMU(
            1,
            2,
            3,
            tf.keras.layers.SimpleRNNCell(4),
            return_sequences=True,
            memory_to_memory=True,
        )(inp)
    elif mode == "fft":
        out = layers.LMUFFT(
            1,
            2,
            3,
            tf.keras.layers.SimpleRNNCell(4),
            return_sequences=True,
        )(inp)

    model = tf.keras.Model(inp, out)

    model.save(str(tmp_path))

    model_load = tf.keras.models.load_model(
        str(tmp_path),
        custom_objects={
            "LMUCell": layers.LMUCell,
            "LMU": layers.LMU,
            "LMUFFT": layers.LMUFFT,
        },
    )

    assert np.allclose(
        model.predict(np.ones((32, 10, 32))), model_load.predict(np.ones((32, 10, 32)))
    )


@pytest.mark.parametrize("return_sequences", (True, False))
@pytest.mark.parametrize(
    "hidden_cell", (None, tf.keras.layers.Dense(4), tf.keras.layers.SimpleRNNCell(4))
)
def test_fft(return_sequences, hidden_cell, rng):
    x = rng.uniform(-1, 1, size=(2, 10, 32))

    rnn_layer = tf.keras.layers.RNN(
        layers.LMUCell(1, 2, 3, hidden_cell),
        return_sequences=return_sequences,
    )
    rnn_out = rnn_layer(x)

    fft_layer = layers.LMUFFT(1, 2, 3, hidden_cell, return_sequences=return_sequences)
    fft_layer.build(x.shape)
    fft_layer.kernel.assign(rnn_layer.cell.kernel)
    fft_out = fft_layer(x)

    assert np.allclose(rnn_out, fft_out, atol=2e-6)


def test_fft_errors():
    fft_layer = layers.LMUFFT(1, 2, 3, None)

    with pytest.raises(ValueError, match="temporal axis be fully specified"):
        fft_layer(tf.keras.Input((None, 32)))


@pytest.mark.parametrize(
    "hidden_to_memory, memory_to_memory, memory_d",
    [(False, False, 1), (True, False, 1), (False, True, 1), (False, False, 2)],
)
def test_fft_auto_swap(hidden_to_memory, memory_to_memory, memory_d):
    lmu = layers.LMU(
        memory_d,
        2,
        3,
        None,
        hidden_to_memory=hidden_to_memory,
        memory_to_memory=memory_to_memory,
    )

    assert (lmu.fft_layer is None) == (
        hidden_to_memory or memory_to_memory or memory_d != 1
    )


@pytest.mark.parametrize(
    "hidden_cell",
    (tf.keras.layers.SimpleRNNCell(units=10), tf.keras.layers.Dense(units=10), None),
)
@pytest.mark.parametrize("fft", (True, False))
def test_hidden_types(hidden_cell, fft, rng, seed):
    x = rng.uniform(-1, 1, size=(2, 5, 32))

    lmu_params = dict(
        memory_d=1,
        order=3,
        theta=4,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
    )

    base_lmu = tf.keras.layers.RNN(
        layers.LMUCell(hidden_cell=None, **lmu_params),
        return_sequences=True,
    )
    base_output = base_lmu(x)
    if isinstance(hidden_cell, tf.keras.layers.SimpleRNNCell):
        base_output = tf.keras.layers.RNN(hidden_cell, return_sequences=True)(
            base_output
        )
    elif isinstance(hidden_cell, tf.keras.layers.Dense):
        base_output = hidden_cell(base_output)

    lmu = (
        layers.LMUFFT(hidden_cell=hidden_cell, return_sequences=True, **lmu_params)
        if fft
        else tf.keras.layers.RNN(
            layers.LMUCell(hidden_cell=hidden_cell, **lmu_params),
            return_sequences=True,
        )
    )
    lmu_output = lmu(x)

    assert np.allclose(lmu_output, base_output, atol=2e-6 if fft else 1e-8)


@pytest.mark.parametrize("fft", (True, False))
def test_connection_params(fft):
    input_shape = (32, 7 if fft else None, 6)
    lmu_args = dict(
        memory_d=1,
        order=3,
        theta=4,
        hidden_cell=tf.keras.layers.Dense(units=5),
        input_to_hidden=False,
    )
    if not fft:
        lmu_args["hidden_to_memory"] = False
        lmu_args["memory_to_memory"] = False

    lmu = layers.LMUCell(**lmu_args) if not fft else layers.LMUFFT(**lmu_args)
    lmu.build(input_shape)
    assert lmu.kernel.shape == (input_shape[-1], lmu.memory_d)
    if not fft:
        assert lmu.recurrent_kernel is None
    assert lmu.hidden_cell.kernel.shape == (
        lmu.memory_d * lmu.order,
        lmu.hidden_cell.units,
    )

    lmu_args["input_to_hidden"] = True
    if not fft:
        lmu_args["hidden_to_memory"] = True
        lmu_args["memory_to_memory"] = True

    lmu = layers.LMUCell(**lmu_args) if not fft else layers.LMUFFT(**lmu_args)
    lmu.hidden_cell.built = False  # so that the kernel will be rebuilt
    lmu.build(input_shape)
    assert lmu.kernel.shape == (
        input_shape[-1] + (lmu.hidden_cell.units if not fft else 0),
        lmu.memory_d,
    )
    if not fft:
        assert lmu.recurrent_kernel.shape == (
            lmu.order * lmu.memory_d,
            lmu.memory_d,
        )
    assert lmu.hidden_cell.kernel.shape == (
        lmu.memory_d * lmu.order + input_shape[-1],
        lmu.hidden_cell.units,
    )


@pytest.mark.parametrize("dropout, recurrent_dropout", [(0, 0), (0.5, 0), (0, 0.5)])
@pytest.mark.parametrize("fft", (True, False))
def test_dropout(dropout, recurrent_dropout, fft):
    if fft:
        kwargs = {}
    else:
        kwargs = dict(memory_to_memory=True, recurrent_dropout=recurrent_dropout)
    lmu = layers.LMU(
        memory_d=1,
        order=3,
        theta=4,
        hidden_cell=None,
        dropout=dropout,
        **kwargs,
    )

    y0 = lmu(np.ones((32, 10, 64)), training=True).numpy()
    y1 = lmu(np.ones((32, 10, 64)), training=True).numpy()

    # if dropout is being applied then outputs should be stochastic, else deterministic
    assert np.allclose(y0, y1) != (dropout or (recurrent_dropout and not fft))

    # dropout not applied when training=False
    y0 = lmu(np.ones((32, 10, 64)), training=False).numpy()
    y1 = lmu(np.ones((32, 10, 64)), training=False).numpy()
    assert np.allclose(y0, y1)


@pytest.mark.parametrize(
    "hidden_cell",
    (tf.keras.layers.SimpleRNNCell(units=10), tf.keras.layers.Dense(units=10), None),
)
def test_skip_connection(rng, hidden_cell):
    memory_d = 4
    order = 16
    n_steps = 10
    input_d = 32

    inp = tf.keras.Input(shape=(n_steps, input_d))

    lmu = layers.LMUCell(
        memory_d=memory_d,
        order=order,
        theta=n_steps,
        hidden_cell=hidden_cell,
        input_to_hidden=True,
    )
    assert lmu.output_size == (None if hidden_cell is None else 10)

    out = tf.keras.layers.RNN(lmu)(inp)

    output_size = (
        (memory_d * order + input_d) if hidden_cell is None else hidden_cell.units
    )
    assert out.shape[-1] == output_size
    assert lmu.hidden_output_size == output_size
    assert lmu.output_size == output_size
