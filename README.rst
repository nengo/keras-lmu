Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks
----------------------------------------------------------------------------------

`Paper <https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks.pdf>`_

We propose a novel memory cell for recurrent neural networks that dynamically maintains information across long windows of time using relatively few resources. The Legendre Memory Unit (LMU) is mathematically derived to orthogonalize its continuous-time history – doing so by solving d coupled ordinary differential equations (ODEs), whose phase space linearly maps onto sliding windows of time via the Legendre polynomials up to degree d − 1 (example d=12, shown below).

.. image:: https://i.imgur.com/Uvl6tj5.png
   :target: https://i.imgur.com/Uvl6tj5.png
   :alt: Legendre polynomials

A single ``LMUCell`` expresses the following computational graph in Keras as an RNN layer, which couples the optimal linear memory (``m``) with a nonlinear hidden state (``h``):

.. image:: https://i.imgur.com/IJGUVg6.png
   :target: https://i.imgur.com/IJGUVg6.png
   :alt: Computational graph

The discretized ``(A, B)`` matrices are initialized according to the LMU's mathematical derivation with respect to some chosen window length, θ. Backpropagation can be used to learn this time-scale, or fine-tune ``(A, B)``, if necessary. By default the coupling between the hidden state (``h``) and the memory vector (``m``) is trained via backpropagation, while the dynamics of the memory remain fixed (`see paper for details <https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks.pdf>`_).

The ``docs`` includes an example for how to use the ``LMUCell``.

The ``paper`` branch in the ``lmu`` GitHub repository includes a pre-trained Keras/TensorFlow model, located at ``models/psMNIST-standard.hdf5``, which obtains the current best-known psMNIST result (using an RNN) of **97.15%**. Note, the network is using fewer internal state-variables and neurons than there are pixels in the input sequence. To reproduce the results from the paper, run the notebooks in the ``experiments`` directory within the ``paper`` branch.

Nengo Examples
--------------

* `Spiking LMUs in Nengo (with online learning) <https://www.nengo.ai/nengo/examples/learning/lmu.html>`_
* `Spiking LMUs in Nengo Loihi (with online learning) <https://www.nengo.ai/nengo-loihi/examples/lmu.html>`_
* `LMUs in NengoDL (reproducing SotA on psMNIST) <https://www.nengo.ai/nengo-dl/examples/lmu.html>`_

Citation
--------

.. code-block::

   @inproceedings{voelker2019lmu,
     title={Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks},
     author={Aaron R. Voelker and Ivana Kaji\'c and Chris Eliasmith},
     booktitle={Advances in Neural Information Processing Systems},
     pages={15544--15553},
     year={2019}
   }
