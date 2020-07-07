.. _basic-usage:

***********
Basic usage
***********

The standard Legendre Memory Unit (LMU) layer
implementation in NengoLMU is defined in the
``lmu.LMU`` class. The following code creates
a new LMU layer:

.. testcode::

   import lmu

   lmu_layer = lmu.LMU(
       units=10,
       order=256,
       theta=784
   )

Note that the values used above for ``units``, ``order``,
and ``theta`` are arbitrary values and the actual values will depend on your
specific solution. ``units`` represents the dimensionality of
the output vector, ``order`` represents the dimensionality of
the memory cell, and ``theta`` represents the dimensionality of
the sliding window. To learn more about these parameters, check out
the :ref:`LMU class API reference <api-reference-lc>`.

Creating NengoLMU layers
------------------------

The ``LMU`` class functions as a standard
TensorFlow layer and is meant to be used within a TensorFlow model.
The code below illustrates how to do this using a TensorFlow functional model with
a 10-dimensional input, and a 20-dimensional output.

.. testcode::

   from tensorflow.keras import Input, Model
   from tensorflow.keras.layers import Dense

   inputs = Input((None, 10))
   lmus = lmu_layer(inputs)
   outputs = Dense(20)(lmus)

   model = Model(inputs=inputs, outputs=outputs)


Customizing parameters
----------------------

The ``LMU`` class is designed to be easy to use and
be integrated seamlessly into your TensorFlow
models. However, for users looking to optimize
their use of the ``LMU`` layer, there are additional
parameters that can be modified.

.. testcode::

    custom_lmu_layer = lmu.LMU(
        units=10,
        order=256,
        theta=784,
        method="zoh",
        hidden_activation="tanh",
    )

The ``method`` parameter specifies the
discretization method that will be used to compute
the ``A`` and ``B`` matrices. By default, this parameter is
set to ``"zoh"`` (zero-order-hold). This is generally the best
option for input signals that are held constant
between time steps, which is a common use case for
sequential data (e.g. feeding in a sequence of
pixels like in the psMNIST task).

The ``hidden_activation`` parameter specifies the
final non-linearity that gets applied to the
output. By default, this parameter is set to
``tanh``. This is mainly done so that outputs
are symmetric about zero and saturated to
``[-1, 1]``, which betters stability. Though,
other non-linearities like ``ReLU`` work well too.

Tuning these parameters can lead to optimized
performance. As an example, using ``"euler"`` as the
discretization method results in an LMU configuration
that is easier to implement on physical hardware and
is also more amenable (produces a more stable system)
in a model where ``theta`` is trained or controlled
on the fly.

Customizing trainability
------------------------

The ``LMU`` class allows users to choose which
encoders and kernels they want to be trained.
By default, every encoder and kernel is
trainable, while the ``A`` and ``B`` matrices
are not.

.. testcode::

    custom_lmu_layer = lmu.LMU(
        units=10,
        order=256,
        theta=784,
        trainable_input_encoders=True,
        trainable_hidden_encoders=True,
        trainable_memory_encoders=True,
        trainable_input_kernel=True,
        trainable_hidden_kernel=True,
        trainable_memory_kernel=True,
        trainable_A=False,
        trainable_B=False,
    )

These trainability flags may be configured however
you would like. The need for specific weights
to be trained will vary on the task being modelled
and the design of the network.

Customizing initializers
------------------------

The ``LMU`` class allows users to customize
the various initializers for the encoders
and kernels. These define the distributions
from which the initial values for the encoder
or kernel weights will be drawn.

.. testcode::

    from tensorflow.keras.initializers import Constant

    custom_lmu_layer = lmu.LMU(
        units=10,
        order=256,
        theta=784,
        input_encoders_initializer=Constant(1),
        hidden_encoders_initializer=Constant(0),
        memory_encoders_initializer=Constant(0),
        input_kernel_initializer=Constant(0),
        hidden_kernel_initializer=Constant(0),
        memory_kernel_initializer="glorot_normal",
    )

These initializers may be configured using
a variety of distributions. Accepted initializers
are listed in the `TensorFlow documentation <https://www.tensorflow.org/api_docs/python/tf/keras/initializers>`_.
Generally, we recommend the ``glorot_uniform``
distribution for feed-forward weights, and an
``orthogonal`` distribution for recurrent weights.
