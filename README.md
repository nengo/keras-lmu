## Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks

We propose a novel memory cell for recurrent neural networks that dynamically maintains information across long windows of time using relatively few resources. The Legendre Memory Unit (LMU) is mathematically derived to orthogonalize its continuous-time history – doing so by solving d coupled ordinary differential equations (ODEs), whose phase space linearly maps onto sliding windows of time via the Legendre polynomials up to degree d − 1 (example d=12, shown below).

![Legendre polynomials](https://i.imgur.com/Uvl6tj5.png)

A single `LMUCell` expresses the following computational graph in Keras as an RNN layer, which couples the optimal linear memory (`m`) with a nonlinear hidden state (`h`):

![Computational graph](https://i.imgur.com/IJGUVg6.png)

The discretized `(A, B)` matrices are initialized according to the LMU's mathematical derivation with respect to some chosen window length, θ. Backpropagation can be used to learn this time-scale, or fine-tune `(A, B)`, if necessary. By default the coupling between the hidden state (`h`) and the memory vector (`m`) is trained via backpropagation, while the dynamics of the memory remain fixed (see paper for details).

This repository includes a pre-trained Keras/TensorFlow model, located at `models/psMNIST-standard.hdf5`, which obtains the current best-known psMNIST result of **97.15%**.

---

We recommend Python 3.6+ and a computer with:

 - At least 32GB of RAM
 - A powerful GPU
 - GPU install of `tensorflow>=1.12.0`

To reproduce all experiments / training / figures:

```
pip install -e .
python figures/legendre-basis.py
python figures/poisson-performance.py
jupyter notebook
```

And then run the notebooks:

 - `experiments/capacity.ipynb`
 - `experiments/psMNIST-standard.ipynb`
 - `experiments/psMNIST-phased-lstm.ipynb`
 - `experiments/mackey-glass.ipynb`

Models will be saved to the `models` directory and figures to the `figures` directory.

A neuromorphic example deployed on Loihi is located at `neuromorphic/loihi_lmu.py`. This requires running `pip install -r neuromorphic/requirements.txt`. If a Loihi board is connected to your computer then it should be automatically detected and used. Otherwise, a detailed hardware emulator is run in software. 

---

Please cite as:
```
@inproceedings{voelker2019lmu,
  title={Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks},
  author={Aaron R. Voelker and Ivana Kaji\'c and Chris Eliasmith},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
