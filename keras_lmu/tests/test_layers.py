# pylint: disable=missing-docstring

import inspect

import keras
import numpy as np
import pytest
import tensorflow as tf
from packaging import version
from scipy.signal import cont2discrete

from keras_lmu import layers


@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
def test_multivariate_lmu(rng, discretizer):
    memory_d = 4
    order = 16
    n_steps = 10 * order
    input_d = 32

    input_enc = rng.uniform(0, 1, size=(input_d, memory_d))

    # check that one multivariate LMU is the same as n one-dimensional LMUs (omitting
    # the hidden part)
    inp = keras.Input(shape=(n_steps, input_d))
    multi_lmu = keras.layers.RNN(
        layers.LMUCell(
            memory_d=memory_d,
            order=order,
            theta=n_steps,
            discretizer=discretizer,
            kernel_initializer=tf.initializers.constant(input_enc),
            hidden_cell=keras.layers.SimpleRNNCell(
                units=memory_d * order,
                activation=None,
                kernel_initializer=tf.initializers.constant(np.eye(memory_d * order)),
                recurrent_initializer=tf.initializers.zeros(),
            ),
        ),
        return_sequences=True,
    )(inp)
    lmus = [
        keras.layers.RNN(
            layers.LMUCell(
                memory_d=1,
                order=order,
                theta=n_steps,
                discretizer=discretizer,
                kernel_initializer=tf.initializers.constant(input_enc[:, [i]]),
                hidden_cell=keras.layers.SimpleRNNCell(
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

    model = keras.Model(inp, [multi_lmu] + lmus)

    results = model.predict(rng.uniform(0, 1, size=(1, n_steps, input_d)))

    for i in range(memory_d):
        assert np.allclose(
            results[0][..., i * order : (i + 1) * order], results[i + 1], atol=2e-5
        ), np.max(abs(results[0][..., i * order : (i + 1) * order] - results[i + 1]))


@pytest.mark.parametrize("has_input_kernel", (True, False))
@pytest.mark.parametrize("feedforward", (True, False))
@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
def test_layer_vs_cell(rng, has_input_kernel, feedforward, discretizer):
    n_steps = 10
    input_d = 32
    kwargs = {
        "memory_d": 4 if has_input_kernel else input_d,
        "order": 12,
        "theta": n_steps * (8 if discretizer == "euler" else 1),
        "discretizer": discretizer,
        "kernel_initializer": "glorot_uniform" if has_input_kernel else None,
        "memory_to_memory": not feedforward,
        "use_bias": True,
    }

    def hidden_cell():
        return keras.layers.SimpleRNNCell(units=64)

    inp = rng.uniform(-1, 1, size=(2, n_steps, input_d))

    lmu_cell = keras.layers.RNN(
        layers.LMUCell(hidden_cell=hidden_cell(), **kwargs),
        return_sequences=True,
    )
    cell_out = lmu_cell(inp)

    lmu_layer = layers.LMU(return_sequences=True, hidden_cell=hidden_cell(), **kwargs)
    lmu_layer.build(inp.shape)
    lmu_layer.layer.set_weights(lmu_cell.get_weights())
    layer_out = lmu_layer(inp)

    assert isinstance(
        lmu_layer.layer, layers.LMUFeedforward if feedforward else keras.layers.RNN
    )

    for w0, w1 in zip(
        sorted(lmu_cell.weights, key=lambda w: w.shape.as_list()),
        sorted(lmu_layer.weights, key=lambda w: w.shape.as_list()),
    ):
        assert np.allclose(w0.numpy(), w1.numpy())

    assert np.allclose(cell_out, lmu_cell(inp))
    assert np.allclose(cell_out, layer_out, atol=3e-6 if feedforward else 1e-8), np.max(
        np.abs(cell_out - layer_out)
    )


@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
@pytest.mark.parametrize("trainable_theta", (True, False))
def test_save_load_weights(rng, tmp_path, discretizer, trainable_theta):
    memory_d = 4
    order = 12
    n_steps = 10
    input_d = 32

    x = rng.uniform(-1, 1, size=(2, n_steps, input_d))

    inp = keras.Input((None, input_d))
    lmu0 = layers.LMU(
        memory_d,
        order,
        n_steps,
        keras.layers.SimpleRNNCell(units=64),
        discretizer=discretizer,
        trainable_theta=trainable_theta,
        return_sequences=True,
    )(inp)
    model0 = keras.Model(inp, lmu0)
    out0 = model0(x)

    lmu1 = layers.LMU(
        memory_d,
        order,
        n_steps,
        keras.layers.SimpleRNNCell(units=64),
        discretizer=discretizer,
        trainable_theta=trainable_theta,
        return_sequences=True,
    )(inp)
    model1 = keras.Model(inp, lmu1)
    out1 = model1(x)

    assert not np.allclose(out0, out1)

    model0.save_weights(tmp_path / "model.weights.h5")
    model1.load_weights(tmp_path / "model.weights.h5")

    out2 = model1(x)
    assert np.allclose(out0, out2)


@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
@pytest.mark.parametrize("trainable_theta", (True, False))
@pytest.mark.parametrize("mode", ("cell", "lmu", "feedforward"))
def test_save_load_serialization(mode, tmp_path, trainable_theta, discretizer):
    if mode == "feedforward" and trainable_theta:
        pytest.skip("LMUFeedforward does not support trainable theta")

    inp = keras.Input((10 if mode == "feedforward" else None, 32))
    if mode == "cell":
        out = keras.layers.RNN(
            layers.LMUCell(
                1,
                2,
                3,
                keras.layers.SimpleRNNCell(4),
                trainable_theta=trainable_theta,
                discretizer=discretizer,
            ),
            return_sequences=True,
        )(inp)
    elif mode == "lmu":
        out = layers.LMU(
            1,
            2,
            3,
            keras.layers.SimpleRNNCell(4),
            return_sequences=True,
            memory_to_memory=True,
            trainable_theta=trainable_theta,
            discretizer=discretizer,
        )(inp)
    elif mode == "feedforward":
        out = layers.LMUFeedforward(
            1,
            2,
            3,
            keras.layers.SimpleRNNCell(4),
            discretizer=discretizer,
            return_sequences=True,
        )(inp)

    model = keras.Model(inp, out)

    model_path = (
        tmp_path
        if version.parse(tf.__version__) < version.parse("2.16.0")
        else tmp_path / "model.keras"
    )
    model.save(model_path)

    model_load = keras.models.load_model(model_path)

    assert np.allclose(
        model.predict(np.ones((32, 10, 32))), model_load.predict(np.ones((32, 10, 32)))
    )


@pytest.mark.parametrize("return_sequences", (True, False))
@pytest.mark.parametrize(
    "hidden_cell",
    (
        lambda: None,
        lambda: keras.layers.Dense(4, dtype="float64"),
        lambda: keras.layers.SimpleRNNCell(4, dtype="float64"),
    ),
)
@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
@pytest.mark.parametrize("conv_mode", ["fft", "raw"])
@pytest.mark.parametrize("seq_len", [10, 50])
def test_feedforward(
    return_sequences, hidden_cell, discretizer, conv_mode, seq_len, rng, seed
):
    kwargs = {
        "memory_d": 4,
        "order": 2,
        "theta": 5,
        "hidden_cell": hidden_cell(),
        "discretizer": discretizer,
        "dtype": "float64",
    }

    x = rng.uniform(-1, 1, size=(2, seq_len, 32))

    rnn_layer = keras.layers.RNN(
        layers.LMUCell(**kwargs),
        return_sequences=return_sequences,
        dtype="float64",
    )
    rnn_out = rnn_layer(x)

    ff_layer = layers.LMUFeedforward(
        return_sequences=return_sequences, conv_mode=conv_mode, **kwargs
    )
    ff_layer.build((2, None, 32))  # testing with unknown sequence length
    ff_layer.set_weights(rnn_layer.get_weights())
    ff_out = ff_layer(x)

    assert ff_out.dtype == rnn_out.dtype == "float64"
    assert np.allclose(
        rnn_out,
        ff_out,
        # larger error for longer sequences, where the error accumulates more
        atol=5e-8 if seq_len < 50 else 5e-4,
    ), np.max(abs(rnn_out - ff_out))


@pytest.mark.parametrize("truncate_ir", [None, 1e-5, 1e-4, 1e-3])
def test_raw_truncation(truncate_ir, rng):
    seq_len = 64
    theta = 11.2
    kwargs = {
        "memory_d": 4,
        "order": 4,
        "theta": theta,
        "hidden_cell": None,
        "kernel_initializer": None,
    }

    x = rng.uniform(-1, 1, size=(2, seq_len, kwargs["memory_d"]))

    rnn_layer = keras.layers.RNN(layers.LMUCell(**kwargs), return_sequences=True)
    rnn_out = rnn_layer(x)

    ff_layer = layers.LMUFeedforward(
        return_sequences=True, conv_mode="raw", truncate_ir=truncate_ir, **kwargs
    )
    ff_out = ff_layer(x)
    if truncate_ir is not None:
        assert ff_layer.impulse_response.shape[0] < seq_len

    # only one seed in 50 failed with `atol=truncate_ir`, hence the 1.2 fudge factor
    atol = 2e-6 if truncate_ir is None else 1.2 * truncate_ir
    assert np.allclose(rnn_out, ff_out, atol=atol)


def test_validation_errors():
    ff_layer = layers.LMUFeedforward(1, 2, 3, None)
    with pytest.warns(UserWarning, match="unknown impulse length"):
        ff_layer(keras.Input((None, 32)))

    with pytest.raises(ValueError, match="hidden_to_memory must be False"):
        layers.LMUCell(1, 2, 3, None, hidden_to_memory=True)

    with pytest.raises(ValueError, match="Unrecognized conv mode"):
        layers.LMUFeedforward(1, 2, 3, None, conv_mode="raw_bad")


@pytest.mark.parametrize(
    "should_use_feedforward, hidden_to_memory, memory_to_memory, trainable_theta",
    [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_feedforward_auto_swap(
    should_use_feedforward, hidden_to_memory, memory_to_memory, trainable_theta
):
    lmu = layers.LMU(
        4,
        2,
        3,
        keras.layers.Dense(5),
        hidden_to_memory=hidden_to_memory,
        memory_to_memory=memory_to_memory,
        trainable_theta=trainable_theta,
    )
    lmu.build((32, None, 8))

    assert isinstance(lmu.layer, layers.LMUFeedforward) == should_use_feedforward


@pytest.mark.parametrize(
    "hidden_cell",
    (keras.layers.SimpleRNNCell(units=10), keras.layers.Dense(units=10), None),
)
@pytest.mark.parametrize("feedforward", (True, False))
def test_hidden_types(hidden_cell, feedforward, rng):
    x = rng.uniform(-1, 1, size=(2, 5, 32))

    lmu_params = {
        "memory_d": 1,
        "order": 3,
        "theta": 4,
        "kernel_initializer": keras.initializers.constant(
            rng.uniform(-1, 1, size=(32, 1))
        ),
    }

    base_lmu = keras.layers.RNN(
        layers.LMUCell(hidden_cell=None, **lmu_params),
        return_sequences=True,
    )
    base_output = base_lmu(x)
    if isinstance(hidden_cell, keras.layers.SimpleRNNCell):
        base_output = keras.layers.RNN(hidden_cell, return_sequences=True)(base_output)
    elif isinstance(hidden_cell, keras.layers.Dense):
        base_output = hidden_cell(base_output)

    lmu = (
        layers.LMUFeedforward(
            hidden_cell=hidden_cell, return_sequences=True, **lmu_params
        )
        if feedforward
        else keras.layers.RNN(
            layers.LMUCell(hidden_cell=hidden_cell, **lmu_params),
            return_sequences=True,
        )
    )
    lmu_output = lmu(x)

    assert np.allclose(
        lmu_output, base_output, atol=3e-6 if feedforward else 1e-8
    ), np.max(np.abs(lmu_output - base_output))


@pytest.mark.parametrize("feedforward", (True, False))
@pytest.mark.parametrize("hidden_cell", (None, keras.layers.Dense))
def test_connection_params(feedforward, hidden_cell):
    input_shape = (32, 7 if feedforward else None, 6)

    x = keras.Input(batch_shape=input_shape)

    lmu_args = {
        "memory_d": 1,
        "order": 3,
        "theta": 4,
        "hidden_cell": hidden_cell if hidden_cell is None else hidden_cell(units=5),
        "input_to_hidden": hidden_cell is not None,
    }

    if not feedforward:
        lmu_args["hidden_to_memory"] = hidden_cell is not None
        lmu_args["memory_to_memory"] = True

    lmu = (
        layers.LMUCell(**lmu_args)
        if not feedforward
        else layers.LMUFeedforward(**lmu_args)
    )
    y = lmu(x) if feedforward else keras.layers.RNN(lmu)(x)
    assert lmu.kernel.shape == (
        input_shape[-1]
        + (0 if feedforward or hidden_cell is None else lmu.hidden_cell.units),
        lmu.memory_d,
    )
    if not feedforward:
        assert lmu.recurrent_kernel.shape == (
            lmu.order * lmu.memory_d,
            lmu.memory_d,
        )
    if hidden_cell is not None:
        assert lmu.hidden_cell.kernel.shape == (
            lmu.memory_d * lmu.order + input_shape[-1],
            lmu.hidden_cell.units,
        )
    assert y.shape == (
        input_shape[0],
        lmu.memory_d * lmu.order if hidden_cell is None else lmu.hidden_cell.units,
    )


@pytest.mark.parametrize(
    "dropout, recurrent_dropout, hidden_dropout, hidden_recurrent_dropout",
    [
        (0, 0, 0, 0),
        (0.5, 0, 0, 0),
        (0, 0.5, 0, 0),
        (0, 0, 0.5, 0),
        pytest.param(
            0,
            0,
            0,
            0.5,
            **(
                {}
                if version.parse(tf.__version__) < version.parse("2.16.0rc0")
                else {
                    "marks": pytest.mark.xfail(
                        reason="TF2.16 recurrent dropout is bugged"
                    )
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize("feedforward", (True, False))
def test_dropout(
    dropout, recurrent_dropout, hidden_dropout, hidden_recurrent_dropout, feedforward
):
    if feedforward:
        kwargs = {}
    else:
        kwargs = {"memory_to_memory": True, "recurrent_dropout": recurrent_dropout}
    lmu = layers.LMU(
        memory_d=1,
        order=3,
        theta=4,
        hidden_cell=keras.layers.SimpleRNNCell(
            5, dropout=hidden_dropout, recurrent_dropout=hidden_recurrent_dropout
        ),
        dropout=dropout,
        **kwargs,
    )

    y0 = lmu(np.ones((32, 10, 64)), training=True).numpy()
    y1 = lmu(np.ones((32, 10, 64)), training=True).numpy()

    # if dropout is being applied then outputs should be stochastic, else deterministic
    assert np.allclose(y0, y1) != (
        dropout > 0
        or (recurrent_dropout > 0 and not feedforward)
        or hidden_dropout > 0
        or hidden_recurrent_dropout > 0
    )

    # dropout not applied when training=False
    y0 = lmu(np.ones((32, 10, 64)), training=False).numpy()
    y1 = lmu(np.ones((32, 10, 64)), training=False).numpy()
    assert np.allclose(y0, y1)


@pytest.mark.parametrize("trainable_theta", (True, False))
@pytest.mark.parametrize("discretizer", ("zoh", "euler"))
@pytest.mark.parametrize("feedforward", (True, False))
def test_fit(feedforward, discretizer, trainable_theta):
    if feedforward and trainable_theta:
        pytest.skip("LMUFeedforward does not support trainable theta")

    lmu_layer = layers.LMU(
        memory_d=1,
        order=256,
        theta=784 if discretizer == "zoh" else 2000,
        trainable_theta=trainable_theta,
        hidden_cell=keras.layers.SimpleRNNCell(units=30),
        hidden_to_memory=not feedforward,
        memory_to_memory=not feedforward,
        input_to_hidden=not feedforward,
        discretizer=discretizer,
        kernel_initializer="zeros",
    )

    inputs = keras.layers.Input((None, 10))
    lmu = lmu_layer(inputs)
    outputs = keras.layers.Dense(2)(lmu)

    model = keras.Model(inputs=inputs, outputs=outputs)

    x_train = tf.ones((5, 5, 10))
    x_test = tf.ones((5, 5, 10))
    y_train = tf.ones((5, 1))
    y_test = tf.ones((5, 1))
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=10, validation_split=0.2)

    _, acc = model.evaluate(x_test, y_test, verbose=0)

    assert isinstance(
        lmu_layer.layer, layers.LMUFeedforward if feedforward else keras.layers.RNN
    )
    assert acc == 1.0


@pytest.mark.parametrize("feedforward", (True, False))
def test_no_input_kernel_dimension_mismatch(feedforward):
    lmu_layer = layers.LMU(
        memory_d=1,
        order=4,
        theta=4,
        hidden_cell=keras.layers.SimpleRNNCell(units=10),
        hidden_to_memory=False,
        memory_to_memory=not feedforward,
        input_to_hidden=not feedforward,
        kernel_initializer=None,
    )

    with pytest.raises(ValueError, match="no input kernel"):
        lmu_layer(tf.ones((4, 10, 2)))


def test_discretizer_types():
    with pytest.raises(ValueError, match="discretizer must be 'zoh' or 'euler'"):
        layers.LMUCell(
            memory_d=1, order=256, theta=784, hidden_cell=None, discretizer="test"
        )


@pytest.mark.parametrize("trainable_theta", (True, False))
def test_discretizer_equivalence(trainable_theta, rng):
    # check that zoh and euler produce approximately the same output
    layer_zoh = layers.LMU(
        memory_d=2,
        order=8,
        theta=256,
        hidden_cell=None,
        discretizer="zoh",
        return_sequences=True,
        kernel_initializer=None,
        trainable_theta=trainable_theta,
    )
    layer_euler = layers.LMU(
        memory_d=2,
        order=8,
        theta=256,
        hidden_cell=None,
        discretizer="euler",
        return_sequences=True,
        kernel_initializer=None,
        trainable_theta=trainable_theta,
    )

    x = rng.uniform(-1, 1, size=(32, 10, 2))

    zoh = layer_zoh(x)
    euler = layer_euler(x)

    assert np.allclose(zoh, euler, atol=0.02), np.max(np.abs(zoh - euler))


def test_cont2discrete_zoh(rng):
    A = rng.randn(64, 64)
    B = rng.randn(64, 1)
    C = np.ones((1, 64))
    D = np.zeros((1,))

    scipy_A, scipy_B, *_ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    tf_A, tf_B = layers.LMUCell._cont2discrete_zoh(A.T, B.T)

    assert np.allclose(scipy_A, tf.transpose(tf_A))
    assert np.allclose(scipy_B, tf.transpose(tf_B))


@pytest.mark.parametrize("discretizer", ("euler", "zoh"))
@pytest.mark.parametrize("trainable_theta", (True, False))
def test_theta_update(discretizer, trainable_theta, tmp_path):
    # create model
    theta = 10
    lmu_cell = layers.LMUCell(
        memory_d=2,
        order=3,
        theta=theta,
        trainable_theta=trainable_theta,
        hidden_cell=keras.layers.SimpleRNNCell(units=4),
        discretizer=discretizer,
    )

    inputs = keras.layers.Input((None, 20))
    lmu = keras.layers.RNN(lmu_cell)(inputs)
    model = keras.Model(inputs=inputs, outputs=lmu)

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer="adam")

    # make sure theta_inv is set correctly to initial value
    assert np.allclose(lmu_cell.theta_inv.numpy(), 1 / theta)

    # fit model on some data
    model.fit(tf.ones((64, 5, 20)), tf.ones((64, 4)), epochs=1)

    # make sure theta kernel got updated if trained
    assert np.allclose(lmu_cell.theta_inv.numpy(), 1 / theta) != trainable_theta

    # save model and make sure you get same outputs, that is, correct theta was stored
    model.save(tmp_path / "model.keras")

    model_load = keras.models.load_model(tmp_path / "model.keras")

    assert np.allclose(
        model.predict(np.ones((32, 10, 20))),
        model_load.predict(np.ones((32, 10, 20))),
    )


@pytest.mark.parametrize("mode", ("cell", "rnn", "feedforward"))
def test_theta_attribute(mode):
    theta = 3

    # check LMUCell theta attribute
    if mode == "cell":
        layer = layers.LMUCell(1, 2, theta, None, trainable_theta=True)
    elif mode == "rnn":
        layer = layers.LMU(1, 2, theta, None, trainable_theta=True)
    elif mode == "feedforward":
        layer = layers.LMU(1, 2, theta, None, trainable_theta=False)

    assert not layer.built
    assert layer.theta == theta

    layer.build((1, 1))
    assert layer.built
    assert np.allclose(layer.theta, theta)

    if mode == "feedforward":
        # feedforward doesn't support trainable theta
        assert isinstance(layer.layer, layers.LMUFeedforward)
    else:
        # check that updates to the internal variable are reflected in the theta
        # attribute
        cell = layer if mode == "cell" else layer.layer.cell
        cell.theta_inv.assign(10)
        assert np.allclose(layer.theta, 0.1)


@pytest.mark.parametrize("fft", (True, False))
@pytest.mark.parametrize("bias", (True, False))
def test_regularizer_loss(fft, bias):
    # choose arbitrary amounts of regularization
    reg = 0.23
    rec_reg = 0.34
    bias_reg = 0.81
    seq_len = 5
    input_d = 3
    memory_d = 2

    lmu_layer = layers.LMU(
        memory_d=memory_d,
        order=4,
        theta=4,
        hidden_cell=keras.layers.SimpleRNNCell(units=10),
        hidden_to_memory=False,
        memory_to_memory=not fft,
        input_to_hidden=not fft,
        use_bias=bias,
        bias_initializer="uniform",  # non-zero to make regularization loss non-zero
        kernel_regularizer=keras.regularizers.L1L2(l1=reg),
        recurrent_regularizer=keras.regularizers.L1L2(l1=rec_reg),
        bias_regularizer=keras.regularizers.L1L2(l1=bias_reg),
    )
    inputs = keras.Input((seq_len, input_d))
    lmus = lmu_layer(inputs)
    outputs = keras.layers.Dense(10)(lmus)

    model = keras.Model(inputs=inputs, outputs=outputs)
    cce_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=cce_loss_fn, optimizer="adam", metrics=[cce_loss_fn])

    n_test = 5
    x_test = tf.ones((n_test, seq_len, input_d))
    y_test = tf.ones((n_test, 1))

    loss, cce_loss = model.evaluate(x_test, y_test)
    regularization_loss = loss - cce_loss

    # determine the target regularization loss based on reg terms and parameter values
    layer = lmu_layer.layer if fft else lmu_layer.layer.cell
    target = reg * np.abs(layer.kernel.numpy()).sum()
    if not fft:
        target += rec_reg * np.abs(layer.recurrent_kernel.numpy()).sum()
    if bias:
        target += bias_reg * np.abs(layer.bias.numpy()).sum()

    assert target > 0.1, f"Target {target} is negligible, please fix the test"
    assert np.isclose(regularization_loss, target)


@pytest.mark.parametrize("cls", [layers.LMUCell, layers.LMU, layers.LMUFeedforward])
def test_get_config(cls):
    """Test that all ``__init__`` arguments appear in the ``get_config`` dictionary."""

    obj = cls(memory_d=2, order=5, theta=3.2, hidden_cell=None)

    config = obj.get_config()
    sig = inspect.signature(cls.__init__)

    for key in sig.parameters:
        if key not in ("self", "kwargs"):
            assert key in config, key


@pytest.mark.parametrize("feedforward", [True, False])
@pytest.mark.parametrize("dtype", [None, "float32", "float64"])
def test_dtype(feedforward, dtype):
    x = np.ones((2, 10, 5))
    layer = layers.LMU(
        1,
        2,
        3,
        keras.layers.SimpleRNNCell(4, dtype=dtype),
        trainable_theta=not feedforward,
        dtype=dtype,
    )
    y = layer(x)
    assert isinstance(
        layer.layer, layers.LMUFeedforward if feedforward else keras.layers.RNN
    )
    assert y.dtype == ("float32" if dtype is None else dtype)
