***************
Getting started
***************

Installation
============

To install NengoLMU, we recommend using ``pip``.

.. code:: bash

   pip install lmu

Before running this command, please ensure your system meets the NengoLMU requirements.

Requirements
------------
NengoLMU has two required dependencies: NumPy and TensorFlow. Because the installation
of these packages is system dependent, it is recommended you follow the instructions
below to install these requirements before installing NengoLMU.

NengoLMU works with Python 3.5 or later.  After installing NumPy and TensorFlow, ``pip``
will do its best to install all of the package's other requirements when it installs
NengoLMU. However, if anything goes wrong during this process, you can install each
required package manually and then try to ``pip install lmu`` again.


Installing NumPy
----------------

.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
.. _Homebrew: https://brew.sh/

The best way to install NumPy depends on several factors, such as your operating system.
If you use Conda_, it is recommended you install NumPy using ``conda install numpy``.
Otherwise, based on your operating system, what we have found works best is:

- Windows: Use the `official installer <https://www.python.org/downloads/>`_ and
  the `unofficial Windows Python binaries <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.
- Mac OS X: Use Homebrew_.
- Linux: Use a package manager or build NumPy from source.


Installing TensorFlow
---------------------

NengoLMU is designed to work within TensorFlow. Assuming you have the required libraries
installed, the latest version of TensorFlow can be using ``pip install tensorflow``

To use TensorFlow with GPU support, you will need to have the CUDA/cuDNN libraries
installed on your system. For this, we recommend you use Conda_
to simplify the installation process. ``conda install tensorflow-gpu`` will install
the TensorFlow package as well as all the CUDA/cuDNN requirements.  If you run into
any problems,  see the `TensorFlow GPU installation instructions <https://www.tensorflow.org/install/gpu>`_
for more details.

In addition to CUDA/cuDNN, TensorFlow's GPU acceleration is only supported with Nvidia GPUs.
Acquiring the appropriate drivers for your Nvidia GPU depends on your system. On Linux, the
correct Nvidia drivers (as of TensorFlow 2.2.0) can be installed via the command
``sudo apt install nvidia-driver-440``. On Windows, Nvidia drivers can be downloaded
from their `website <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_.

It is also possible to build TensorFlow from source. This is significantly
more complicated but allows you to customize the installation to your specific
system configuration, which can improve simulation speeds. See the system specific
instructions below:

* `Instructions for installing on Windows
  <https://www.tensorflow.org/install/source_windows>`_.

* `Instructions for installing TensorFlow on Ubuntu or Mac OS
  <https://www.tensorflow.org/install/source>`_.


Installing other packages
-------------------------

The steps above will only install NengoLMU's required dependencies.
Optional NengoLMU features require additional packages to be installed.

- An optional Legendre initializer
  requires SciPy. You can read more about
  this initializer in the :ref:`API reference <api-reference-li>`.
- Running the test suite requires pytest.
- Building the documentation requires Sphinx, NumPyDoc, nengo_sphinx_theme,
  and a few other packages.

These additional dependencies can also be installed through ``pip`` when
installing NengoLMU.

.. code-block:: bash

   pip install lmu[optional]  # Needed to use the Legendre initializer
   pip install lmu[tests]  # Needed to run unit tests
   pip install lmu[docs]  # Needed to build docs
   pip install lmu[all]  # All of the above

Next steps
==========

* If you want to learn how to use NengoLMU in your
  models, read through the :ref:`basic usage <basic-usage>` page.
* For a more detailed understanding of the various
  classes and functions in the NengoLMU package,
  refer to the :ref:`API reference <api-reference>`.
* If you are interested to learn the theoretical background
  behind how the Legendre Memory Unit works, we recommend reading
  `this technical overview <http://compneuro.uwaterloo.ca/files/publications/voelker.2019.lmu.pdf>`_.
* If you would like to see how NengoLMU is incorporated into various
  models, check out our :ref:`examples <examples>`.
