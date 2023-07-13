"""Core classes for the KerasLMU package."""

import warnings

import numpy as np
import tensorflow as tf
from packaging import version

# pylint: disable=ungrouped-imports
if version.parse(tf.__version__) < version.parse("2.6.0rc0"):
    from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
elif version.parse(tf.__version__) < version.parse("2.9.0rc0"):
    from keras.layers.recurrent import DropoutRNNCellMixin
elif version.parse(tf.__version__) < version.parse("2.13.0rc0"):
    from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
else:
    from keras.src.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin

if version.parse(tf.__version__) < version.parse("2.8.0rc0"):
    from tensorflow.keras.layers import Layer as BaseRandomLayer
elif version.parse(tf.__version__) < version.parse("2.13.0rc0"):
    from keras.engine.base_layer import BaseRandomLayer
else:
    from keras.src.engine.base_layer import BaseRandomLayer


@tf.keras.utils.register_keras_serializable("keras-lmu")
class LMUCell(DropoutRNNCellMixin, BaseRandomLayer):
    """
    Implementation of LMU cell (to be used within Keras RNN wrapper).

    In general, the LMU cell consists of two parts: a memory component (decomposing
    the input signal using Legendre polynomials as a basis), and a hidden component
    (learning nonlinear mappings from the memory component). [1]_ [2]_

    This class processes one step within the whole time sequence input. Use the ``LMU``
    class to create a recurrent Keras layer to process the whole sequence. Calling
    ``LMU()`` is equivalent to doing ``RNN(LMUCell())``.

    Parameters
    ----------
    memory_d : int
        Dimensionality of input to memory component.
    order : int
        The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. This parameter sets the number of
        Legendre polynomials used to orthogonally represent the sliding window.
    theta : float
        The number of timesteps in the sliding window that is represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented at the time of prediction, however the
        entire sequence will still be processed in order for information to be
        projected to and from the hidden layer. If ``trainable_theta`` is enabled, then
        theta will be updated during the course of training.
    hidden_cell : ``tf.keras.layers.Layer``
        Keras Layer/RNNCell implementing the hidden component.
    trainable_theta : bool
        If True, theta is learnt over the course of training. Otherwise, it is kept
        constant.
    hidden_to_memory : bool
        If True, connect the output of the hidden component back to the memory
        component (default False).
    memory_to_memory : bool
        If True, add a learnable recurrent connection (in addition to the static
        Legendre system) to the memory component (default False).
    input_to_hidden : bool
        If True, connect the input directly to the hidden component (in addition to
        the connection from the memory component) (default False).
    discretizer : str
        The method used to discretize the A and B matrices of the LMU. Current
        options are "zoh" (short for Zero Order Hold) and "euler".
        "zoh" is more accurate, but training will be slower than "euler" if
        ``trainable_theta=True``. Note that a larger theta is needed when discretizing
        using "euler" (a value that is larger than ``4*order`` is recommended).
    kernel_initializer : ``tf.initializers.Initializer``
        Initializer for weights from input to memory/hidden component. If ``None``,
        no weights will be used, and the input size must match the memory/hidden size.
    recurrent_initializer : ``tf.initializers.Initializer``
        Initializer for ``memory_to_memory`` weights (if that connection is enabled).
    kernel_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for weights from input to memory/hidden component.
    recurrent_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for ``memory_to_memory`` weights (if that connection is enabled).
    use_bias : bool
        If True, the memory component includes a bias term.
    bias_initializer : ``tf.initializers.Initializer``
        Initializer for the memory component bias term. Only used if ``use_bias=True``.
    bias_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for the memory component bias term. Only used if ``use_bias=True``.
    dropout : float
        Dropout rate on input connections.
    recurrent_dropout : float
        Dropout rate on ``memory_to_memory`` connection.

    References
    ----------
    .. [1] Voelker and Eliasmith (2018). Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells.
       Neural Computation, 30(3): 569-609.
    .. [2] Voelker and Eliasmith. "Methods and systems for implementing
       dynamic neural networks." U.S. Patent Application No. 15/243,223.
       Filing date: 2016-08-22.
    """

    def __init__(
        self,
        memory_d,
        order,
        theta,
        hidden_cell,
        trainable_theta=False,
        hidden_to_memory=False,
        memory_to_memory=False,
        input_to_hidden=False,
        discretizer="zoh",
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        use_bias=False,
        bias_initializer="zeros",
        bias_regularizer=None,
        dropout=0,
        recurrent_dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.memory_d = memory_d
        self.order = order
        self._init_theta = theta
        self.hidden_cell = hidden_cell
        self.trainable_theta = trainable_theta
        self.hidden_to_memory = hidden_to_memory
        self.memory_to_memory = memory_to_memory
        self.input_to_hidden = input_to_hidden
        self.discretizer = discretizer
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.theta_inv = None
        self.A = None
        self.B = None

        if self.discretizer not in ("zoh", "euler"):
            raise ValueError(
                f"discretizer must be 'zoh' or 'euler' (got '{self.discretizer}')"
            )

        if self.hidden_cell is None:
            if self.hidden_to_memory:
                raise ValueError(
                    "hidden_to_memory must be False if hidden_cell is None"
                )

            self.hidden_output_size = self.memory_d * self.order
            self.hidden_state_size = []
        elif hasattr(self.hidden_cell, "state_size"):
            self.hidden_output_size = self.hidden_cell.output_size
            self.hidden_state_size = self.hidden_cell.state_size
        else:
            # TODO: support layers that don't have the `units` attribute
            self.hidden_output_size = self.hidden_cell.units
            self.hidden_state_size = [self.hidden_cell.units]

        self.state_size = [self.memory_d * self.order] + tf.nest.flatten(
            self.hidden_state_size
        )
        self.output_size = self.hidden_output_size

    @property
    def theta(self):
        """
        Value of the ``theta`` parameter.

        If ``trainable_theta=True`` this returns the trained value, not the
        initial value passed in to the constructor.
        """
        if self.built:
            return 1 / tf.keras.backend.get_value(self.theta_inv)

        return self._init_theta

    def _gen_AB(self):
        """Generates A and B matrices."""

        # compute analog A/B matrices
        Q = np.arange(self.order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        # discretize matrices
        if self.discretizer == "zoh":
            # save the un-discretized matrices for use in .call
            self._base_A = tf.constant(A.T, dtype=self.dtype)
            self._base_B = tf.constant(B.T, dtype=self.dtype)

            self.A, self.B = LMUCell._cont2discrete_zoh(
                self._base_A / self._init_theta, self._base_B / self._init_theta
            )
        else:
            if not self.trainable_theta:
                A = A / self._init_theta + np.eye(self.order)
                B = B / self._init_theta

            self.A = tf.constant(A.T, dtype=self.dtype)
            self.B = tf.constant(B.T, dtype=self.dtype)

    @staticmethod
    def _cont2discrete_zoh(A, B):
        """
        Function to discretize A and B matrices using Zero Order Hold method.

        Functionally equivalent to
        ``scipy.signal.cont2discrete((A.T, B.T, _, _), method="zoh", dt=1.0)``
        (but implemented in TensorFlow so that it is differentiable).

        Note that this accepts and returns matrices that are transposed from the
        standard linear system implementation (as that makes it easier to use in
        `.call`).
        """

        # combine A/B and pad to make square matrix
        em_upper = tf.concat([A, B], axis=0)  # pylint: disable=no-value-for-parameter
        em = tf.pad(em_upper, [(0, 0), (0, B.shape[0])])

        # compute matrix exponential
        ms = tf.linalg.expm(em)

        # slice A/B back out of combined matrix
        discrt_A = ms[: A.shape[0], : A.shape[1]]
        discrt_B = ms[A.shape[0] :, : A.shape[1]]

        return discrt_A, discrt_B

    def build(self, input_shape):
        """
        Builds the cell.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shape)

        enc_d = input_shape[-1]
        if self.hidden_to_memory:
            enc_d += self.hidden_output_size

        if self.kernel_initializer is not None:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(enc_d, self.memory_d),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
            )
        elif enc_d != self.memory_d:
            raise ValueError(
                f"For LMUCells with no input kernel, the input dimension ({enc_d})"
                f" must equal `memory_d` ({self.memory_d})."
            )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.memory_d,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
            )

        # when using euler, 1/theta results in better gradients for the memory
        # update since you are multiplying 1/theta, as compared to dividing theta
        if self.trainable_theta:
            self.theta_inv = self.add_weight(
                name="theta_inv",
                shape=(),
                initializer=tf.initializers.constant(1 / self._init_theta),
                constraint=tf.keras.constraints.NonNeg(),
            )
        else:
            self.theta_inv = tf.constant(1 / self._init_theta, dtype=self.dtype)

        if self.memory_to_memory:
            self.recurrent_kernel = self.add_weight(
                name="recurrent_kernel",
                shape=(self.memory_d * self.order, self.memory_d),
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
            )
        else:
            self.recurrent_kernel = None

        if self.hidden_cell is not None and not self.hidden_cell.built:
            hidden_input_d = self.memory_d * self.order
            if self.input_to_hidden:
                hidden_input_d += input_shape[-1]
            with tf.name_scope(self.hidden_cell.name):
                self.hidden_cell.build((input_shape[0], hidden_input_d))

        # generate A and B matrices
        self._gen_AB()

    def call(self, inputs, states, training=None):  # noqa: C901
        """
        Apply this cell to inputs.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        if training is None:
            training = tf.keras.backend.learning_phase()

        states = tf.nest.flatten(states)

        # state for the LMU memory
        m = states[0]
        # state for the hidden cell
        h = states[1:]

        # compute memory input
        u = (
            tf.concat((inputs, h[0]), axis=1)  # pylint: disable=no-value-for-parameter
            if self.hidden_to_memory
            else inputs
        )
        if self.dropout > 0:
            u *= self.get_dropout_mask_for_cell(u, training)
        if self.kernel is not None:
            u = tf.matmul(u, self.kernel, name="kernel_matmul")
        if self.bias is not None:
            u = u + self.bias

        if self.memory_to_memory:
            if self.recurrent_dropout > 0:
                # note: we don't apply dropout to the memory input, only
                # the recurrent kernel
                rec_m = m * self.get_recurrent_dropout_mask_for_cell(m, training)
            else:
                rec_m = m

            u = u + tf.matmul(
                rec_m, self.recurrent_kernel, name="recurrent_kernel_matmul"
            )

        # separate memory/order dimensions
        m = tf.reshape(m, (-1, self.memory_d, self.order))
        u = tf.expand_dims(u, -1)

        # update memory
        if self.discretizer == "zoh" and self.trainable_theta:
            # apply updated theta and re-discretize
            A, B = LMUCell._cont2discrete_zoh(
                self._base_A * self.theta_inv, self._base_B * self.theta_inv
            )
        else:
            A, B = self.A, self.B

        _m = tf.matmul(m, A) + tf.matmul(u, B)

        if self.discretizer == "euler" and self.trainable_theta:
            # apply updated theta. this is the same as scaling A/B by theta, but it's
            # more efficient to do it this way.
            # note that when computing this way the A matrix does not
            # include the identity matrix along the diagonal (since we don't want to
            # scale that part by theta), which is why we do += instead of =
            m += _m * self.theta_inv
        else:
            m = _m

        # re-combine memory/order dimensions
        m = tf.reshape(m, (-1, self.memory_d * self.order))

        # apply hidden cell
        h_in = (
            tf.concat((m, inputs), axis=1)  # pylint: disable=no-value-for-parameter
            if self.input_to_hidden
            else m
        )

        if self.hidden_cell is None:
            o = h_in
            h = []
        elif hasattr(self.hidden_cell, "state_size"):
            o, h = self.hidden_cell(h_in, h, training=training)
        else:
            o = self.hidden_cell(h_in, training=training)
            h = [o]

        return o, [m] + h

    def reset_dropout_mask(self):
        """Reset dropout mask for memory and hidden components."""
        super().reset_dropout_mask()
        if isinstance(self.hidden_cell, DropoutRNNCellMixin):
            self.hidden_cell.reset_dropout_mask()

    def reset_recurrent_dropout_mask(self):
        """Reset recurrent dropout mask for memory and hidden components."""
        super().reset_recurrent_dropout_mask()
        if isinstance(self.hidden_cell, DropoutRNNCellMixin):
            self.hidden_cell.reset_recurrent_dropout_mask()

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update(
            {
                "memory_d": self.memory_d,
                "order": self.order,
                "theta": self._init_theta,
                "hidden_cell": tf.keras.layers.serialize(self.hidden_cell),
                "trainable_theta": self.trainable_theta,
                "hidden_to_memory": self.hidden_to_memory,
                "memory_to_memory": self.memory_to_memory,
                "input_to_hidden": self.input_to_hidden,
                "discretizer": self.discretizer,
                "kernel_initializer": self.kernel_initializer,
                "recurrent_initializer": self.recurrent_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "recurrent_regularizer": self.recurrent_regularizer,
                "use_bias": self.use_bias,
                "bias_initializer": self.bias_initializer,
                "bias_regularizer": self.bias_regularizer,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hidden_cell"] = (
            None
            if config["hidden_cell"] is None
            else tf.keras.layers.deserialize(config["hidden_cell"])
        )
        return super().from_config(config)


@tf.keras.utils.register_keras_serializable("keras-lmu")
class LMU(tf.keras.layers.Layer):
    """
    A layer of trainable low-dimensional delay systems.

    Each unit buffers its encoded input
    by internally representing a low-dimensional
    (i.e., compressed) version of the sliding window.

    Nonlinear decodings of this representation,
    expressed by the A and B matrices, provide
    computations across the window, such as its
    derivative, energy, median value, etc ([1]_, [2]_).
    Note that these decoder matrices can span across
    all of the units of an input sequence.

    Parameters
    ----------
    memory_d : int
        Dimensionality of input to memory component.
    order : int
        The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. This parameter sets the number of
        Legendre polynomials used to orthogonally represent the sliding window.
    theta : float
        The number of timesteps in the sliding window that is represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented at the time of prediction, however the
        entire sequence will still be processed in order for information to be
        projected to and from the hidden layer. If ``trainable_theta`` is enabled, then
        theta will be updated during the course of training.
    hidden_cell : ``tf.keras.layers.Layer``
        Keras Layer/RNNCell implementing the hidden component.
    trainable_theta : bool
        If True, theta is learnt over the course of training. Otherwise, it is kept
        constant.
    hidden_to_memory : bool
        If True, connect the output of the hidden component back to the memory
        component (default False).
    memory_to_memory : bool
        If True, add a learnable recurrent connection (in addition to the static
        Legendre system) to the memory component (default False).
    input_to_hidden : bool
        If True, connect the input directly to the hidden component (in addition to
        the connection from the memory component) (default False).
    discretizer : str
        The method used to discretize the A and B matrices of the LMU. Current
        options are "zoh" (short for Zero Order Hold) and "euler".
        "zoh" is more accurate, but training will be slower than "euler" if
        ``trainable_theta=True``. Note that a larger theta is needed when discretizing
        using "euler" (a value that is larger than ``4*order`` is recommended).
    kernel_initializer : ``tf.initializers.Initializer``
        Initializer for weights from input to memory/hidden component. If ``None``,
        no weights will be used, and the input size must match the memory/hidden size.
    recurrent_initializer : ``tf.initializers.Initializer``
        Initializer for ``memory_to_memory`` weights (if that connection is enabled).
    kernel_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for weights from input to memory/hidden component.
    recurrent_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for ``memory_to_memory`` weights (if that connection is enabled).
    use_bias : bool
        If True, the memory component includes a bias term.
    bias_initializer : ``tf.initializers.Initializer``
        Initializer for the memory component bias term. Only used if ``use_bias=True``.
    bias_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for the memory component bias term. Only used if ``use_bias=True``.
    dropout : float
        Dropout rate on input connections.
    recurrent_dropout : float
        Dropout rate on ``memory_to_memory`` connection.
    return_sequences : bool, optional
        If True, return the full output sequence. Otherwise, return just the last
        output in the output sequence.

    References
    ----------
    .. [1] Voelker and Eliasmith (2018). Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells.
       Neural Computation, 30(3): 569-609.
    .. [2] Voelker and Eliasmith. "Methods and systems for implementing
       dynamic neural networks." U.S. Patent Application No. 15/243,223.
       Filing date: 2016-08-22.
    """

    def __init__(
        self,
        memory_d,
        order,
        theta,
        hidden_cell,
        trainable_theta=False,
        hidden_to_memory=False,
        memory_to_memory=False,
        input_to_hidden=False,
        discretizer="zoh",
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        use_bias=False,
        bias_initializer="zeros",
        bias_regularizer=None,
        dropout=0,
        recurrent_dropout=0,
        return_sequences=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.memory_d = memory_d
        self.order = order
        self._init_theta = theta
        self.hidden_cell = hidden_cell
        self.trainable_theta = trainable_theta
        self.hidden_to_memory = hidden_to_memory
        self.memory_to_memory = memory_to_memory
        self.input_to_hidden = input_to_hidden
        self.discretizer = discretizer
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.layer = None

    @property
    def theta(self):
        """
        Value of the ``theta`` parameter.

        If ``trainable_theta=True`` this returns the trained value, not the
        initial value passed in to the constructor.
        """

        if self.built:
            return (
                self.layer.theta
                if isinstance(self.layer, LMUFeedforward)
                else self.layer.cell.theta
            )

        return self._init_theta

    def build(self, input_shape):
        """
        Builds the layer.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shape)

        if (
            not self.hidden_to_memory
            and not self.memory_to_memory
            and not self.trainable_theta
        ):
            self.layer = LMUFeedforward(
                memory_d=self.memory_d,
                order=self.order,
                theta=self._init_theta,
                hidden_cell=self.hidden_cell,
                input_to_hidden=self.input_to_hidden,
                discretizer=self.discretizer,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                dropout=self.dropout,
                return_sequences=self.return_sequences,
                dtype=self.dtype,
            )
        else:
            self.layer = tf.keras.layers.RNN(
                LMUCell(
                    memory_d=self.memory_d,
                    order=self.order,
                    theta=self._init_theta,
                    hidden_cell=self.hidden_cell,
                    trainable_theta=self.trainable_theta,
                    hidden_to_memory=self.hidden_to_memory,
                    memory_to_memory=self.memory_to_memory,
                    input_to_hidden=self.input_to_hidden,
                    discretizer=self.discretizer,
                    kernel_initializer=self.kernel_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    recurrent_regularizer=self.recurrent_regularizer,
                    use_bias=self.use_bias,
                    bias_initializer=self.bias_initializer,
                    bias_regularizer=self.bias_regularizer,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    dtype=self.dtype,
                ),
                return_sequences=self.return_sequences,
                dtype=self.dtype,
            )

        self.layer.build(input_shape)

    def call(self, inputs, training=None):
        """
        Apply this layer to inputs.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        return self.layer.call(inputs, training=training)

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update(
            {
                "memory_d": self.memory_d,
                "order": self.order,
                "theta": self._init_theta,
                "hidden_cell": tf.keras.layers.serialize(self.hidden_cell),
                "trainable_theta": self.trainable_theta,
                "hidden_to_memory": self.hidden_to_memory,
                "memory_to_memory": self.memory_to_memory,
                "input_to_hidden": self.input_to_hidden,
                "discretizer": self.discretizer,
                "kernel_initializer": self.kernel_initializer,
                "recurrent_initializer": self.recurrent_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "recurrent_regularizer": self.recurrent_regularizer,
                "use_bias": self.use_bias,
                "bias_initializer": self.bias_initializer,
                "bias_regularizer": self.bias_regularizer,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
                "return_sequences": self.return_sequences,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hidden_cell"] = (
            None
            if config["hidden_cell"] is None
            else tf.keras.layers.deserialize(config["hidden_cell"])
        )
        return super().from_config(config)


@tf.keras.utils.register_keras_serializable("keras-lmu")
class LMUFeedforward(tf.keras.layers.Layer):
    """
    Layer class for the feedforward variant of the LMU.

    This class assumes no recurrent connections are desired in the memory component.

    Produces the output of the delay system by evaluating the convolution of the input
    sequence with the impulse response from the LMU cell.

    Parameters
    ----------
    memory_d : int
        Dimensionality of input to memory component.
    order : int
        The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. This parameter sets the number of
        Legendre polynomials used to orthogonally represent the sliding window.
    theta : float
        The number of timesteps in the sliding window that is represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented at the time of prediction, however the
        entire sequence will still be processed in order for information to be
        projected to and from the hidden layer.
    hidden_cell : ``tf.keras.layers.Layer``
        Keras Layer implementing the hidden component.
    input_to_hidden : bool
        If True, connect the input directly to the hidden component (in addition to
        the connection from the memory component) (default False).
    discretizer : str
        The method used to discretize the A and B matrices of the LMU. Current
        options are "zoh" (short for Zero Order Hold) and "euler".
        "zoh" is more accurate, but training will be slower than "euler" if
        ``trainable_theta=True``. Note that a larger theta is needed when discretizing
        using "euler" (a value that is larger than ``4*order`` is recommended).
    kernel_initializer : ``tf.initializers.Initializer``
        Initializer for weights from input to memory/hidden component. If ``None``,
        no weights will be used, and the input size must match the memory/hidden size.
    kernel_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for weights from input to memory/hidden component.
    use_bias : bool
        If True, the memory component includes a bias term.
    bias_initializer : ``tf.initializers.Initializer``
        Initializer for the memory component bias term. Only used if ``use_bias=True``.
    bias_regularizer : ``tf.keras.regularizers.Regularizer``
        Regularizer for the memory component bias term. Only used if ``use_bias=True``.
    dropout : float
        Dropout rate on input connections.
    return_sequences : bool, optional
        If True, return the full output sequence. Otherwise, return just the last
        output in the output sequence.
    conv_mode : "fft" or "raw"
        The method for performing the inpulse response convolution. "fft" uses FFT
        convolution (default). "raw" uses explicit convolution, which may be faster
        for particular models on particular hardware.
    truncate_ir : float
        The portion of the impulse response to truncate when using "raw"
        convolution (see ``conv_mode``). This is an approximate upper bound on the error
        relative to the exact implementation. Smaller ``theta`` values result in more
        truncated elements for a given value of ``truncate_ir``, improving efficiency.
    """

    def __init__(
        self,
        memory_d,
        order,
        theta,
        hidden_cell,
        input_to_hidden=False,
        discretizer="zoh",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_bias=False,
        bias_initializer="zeros",
        bias_regularizer=None,
        dropout=0,
        return_sequences=False,
        conv_mode="fft",
        truncate_ir=1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if conv_mode not in ("fft", "raw"):
            raise ValueError(f"Unrecognized conv mode '{conv_mode}'")

        self.memory_d = memory_d
        self.order = order
        self.theta = theta
        self.hidden_cell = hidden_cell
        self.input_to_hidden = input_to_hidden
        self.discretizer = discretizer
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.conv_mode = conv_mode.lower()
        self.truncate_ir = truncate_ir

        # create a standard LMUCell to generate the impulse response during `build`
        self.delay_layer = tf.keras.layers.RNN(
            LMUCell(
                memory_d=1,
                order=order,
                theta=theta,
                hidden_cell=None,
                trainable_theta=False,
                input_to_hidden=False,
                hidden_to_memory=False,
                memory_to_memory=False,
                discretizer=discretizer,
                kernel_initializer=None,
                trainable=False,
                dtype=self.dtype,
            ),
            return_sequences=True,
            dtype=self.dtype,
        )
        self.impulse_response = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        """
        Builds the layer.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shape)

        enc_d = input_shape[-1]
        seq_len = input_shape[1]
        if seq_len is None:
            theta_factor = 5
            warnings.warn(
                f"Approximating unknown impulse length with {theta_factor}*theta; "
                f"setting a fixed sequence length on inputs will remove the need for "
                f"approximation"
            )
            impulse_len = self.theta * theta_factor
        else:
            impulse_len = seq_len

        impulse = tf.reshape(tf.eye(impulse_len, 1), (1, -1, 1))

        self.impulse_response = tf.squeeze(
            self.delay_layer(impulse, training=False), axis=0
        )

        if self.conv_mode == "fft":
            self.impulse_response_fft = (
                None
                if seq_len is None
                else tf.signal.rfft(
                    tf.transpose(self.impulse_response),
                    fft_length=[2 * seq_len],
                )
            )
        else:
            if self.truncate_ir is not None:
                assert self.impulse_response.shape == (impulse_len, self.order)

                cumsum = tf.math.cumsum(
                    tf.math.abs(self.impulse_response), axis=0, reverse=True
                )
                cumsum = cumsum / cumsum[0]
                to_drop = tf.reduce_all(cumsum < self.truncate_ir, axis=-1)
                if to_drop[-1]:
                    cutoff = tf.where(to_drop)[0, -1]
                    self.impulse_response = self.impulse_response[:cutoff]

            self.impulse_response = tf.reshape(
                self.impulse_response,
                (self.impulse_response.shape[0], 1, 1, self.order),
            )
            self.impulse_response = self.impulse_response[::-1, :, :, :]

        if self.kernel_initializer is not None:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(input_shape[-1], self.memory_d),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
            )
        else:
            self.kernel = None
            if enc_d != self.memory_d:
                raise ValueError(
                    f"For LMUCells with no input kernel, the input dimension ({enc_d})"
                    f" must equal `memory_d` ({self.memory_d})."
                )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.memory_d,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
            )

        if self.hidden_cell is not None and not self.hidden_cell.built:
            hidden_input_d = self.memory_d * self.order
            if self.input_to_hidden:
                hidden_input_d += input_shape[-1]
            with tf.name_scope(self.hidden_cell.name):
                self.hidden_cell.build((input_shape[0], hidden_input_d))

    def call(self, inputs, training=None):
        """
        Apply this layer to inputs.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        if training is None:
            training = tf.keras.backend.learning_phase()

        if self.dropout:
            inputs = tf.keras.layers.Dropout(
                self.dropout, noise_shape=(inputs.shape[0], 1) + inputs.shape[2:]
            )(inputs)

        # Apply input encoders
        u = inputs
        if self.kernel is not None:
            u = tf.matmul(u, self.kernel, name="kernel_matmul")
        if self.bias is not None:
            u = u + self.bias

        if self.conv_mode == "fft":
            m = self._fft_convolution(u)
        elif self.conv_mode == "raw":
            m = self._raw_convolution(u)

        # apply hidden cell
        h_in = (
            tf.concat((m, inputs), axis=-1)  # pylint: disable=no-value-for-parameter
            if self.input_to_hidden
            else m
        )

        if self.hidden_cell is None:
            h = h_in if self.return_sequences else h_in[:, -1]
        elif hasattr(self.hidden_cell, "state_size"):
            h = tf.keras.layers.RNN(
                self.hidden_cell,
                return_sequences=self.return_sequences,
                dtype=self.dtype,
            )(h_in, training=training)
        else:
            if not self.return_sequences:
                # no point applying the hidden cell to the whole sequence
                h = self.hidden_cell(h_in[:, -1], training=training)
            else:
                h = tf.keras.layers.TimeDistributed(self.hidden_cell)(
                    h_in, training=training
                )

        return h

    def _fft_convolution(self, u):
        seq_len = tf.shape(u)[1]

        # FFT requires shape (batch, memory_d, timesteps)
        u = tf.transpose(u, perm=[0, 2, 1])

        # Pad sequences to avoid circular convolution
        # Perform the FFT
        fft_input = tf.signal.rfft(u, fft_length=[2 * seq_len])
        impulse_response = (
            tf.signal.rfft(
                tf.transpose(self.impulse_response[:seq_len]),
                fft_length=[2 * seq_len],
            )
            if self.impulse_response_fft is None
            else self.impulse_response_fft
        )

        # Elementwise product of FFT (with broadcasting)
        result = tf.expand_dims(fft_input, axis=-2) * impulse_response

        # Inverse FFT
        m = tf.signal.irfft(result, fft_length=[2 * seq_len])[..., :seq_len]

        m = tf.reshape(m, (-1, self.order * self.memory_d, seq_len))

        return tf.transpose(m, perm=[0, 2, 1])

    def _raw_convolution(self, u):
        seq_len = tf.shape(u)[1]
        ir_len = self.impulse_response.shape[0]

        u = tf.expand_dims(u, -1)
        m = tf.nn.conv2d(
            u,
            self.impulse_response,
            strides=1,
            data_format="NHWC",
            padding=[[0, 0], [ir_len - 1, 0], [0, 0], [0, 0]],
        )
        m = tf.reshape(m, (-1, seq_len, self.memory_d * self.order))
        return m

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update(
            {
                "memory_d": self.memory_d,
                "order": self.order,
                "theta": self.theta,
                "hidden_cell": tf.keras.layers.serialize(self.hidden_cell),
                "input_to_hidden": self.input_to_hidden,
                "discretizer": self.discretizer,
                "kernel_initializer": self.kernel_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "use_bias": self.use_bias,
                "bias_initializer": self.bias_initializer,
                "bias_regularizer": self.bias_regularizer,
                "dropout": self.dropout,
                "return_sequences": self.return_sequences,
                "conv_mode": self.conv_mode,
                "truncate_ir": self.truncate_ir,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hidden_cell"] = (
            None
            if config["hidden_cell"] is None
            else tf.keras.layers.deserialize(config["hidden_cell"])
        )
        return super().from_config(config)
