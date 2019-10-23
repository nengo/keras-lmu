# neurips2019
Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks

We recommend Python 3.6+ and a computer with:

 - At least 32GB of RAM
 - A powerful GPU
 - GPU install of `tensorflow>=1.12.0`

The Keras model `models/psMNIST-standard.hdf5` reproduces the psMNIST test result of 97.15%.

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
 - `experiments/mackey-glass.ipynb`

Models will be saved to the `models` directory and figures to the `figures` directory.
