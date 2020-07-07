************************
Contributing to NengoLMU
************************

Please read our
`general contributor guide <https://www.nengo.ai/contributing/>`_
first.
The instructions below specifically apply
to the NengoLMU project.

.. _dev-install:

Developer installation
======================

If you want to change parts of NengoLMU,
you should do a developer installation,
and install all of the optional dependencies.

.. code-block:: bash

   git clone https://github.com/abr/lmu.git
   cd lmu
   pip install -e .

How to run unit tests
=====================

NengoLMU contains a test suite, which we run with pytest_.
To run these tests do

.. code-block:: bash

   pytest --pyargs lmu

Running individual tests
------------------------

Tests in a specific test file can be run by calling
``pytest`` on that file. For example

.. code-block:: bash

   pytest lmu/tests/test_cell.py

will run all the tests in ``test_cell.py``.

Individual tests can be run using the ``-k EXPRESSION`` argument. Only tests
that match the given substring expression are run. For example

.. code-block:: bash

   pytest lmu/tests/test_cell.py -k abc

will run any tests with ``abc`` in the name, in the file
``test_cell.py``.

Getting help and other options
------------------------------

Information about ``pytest`` usage
and NengoLMU-specific options can be found with

.. code-block:: bash

   pytest --pyargs lmu --help

Writing your own tests
----------------------

When writing your own tests, please make use of
custom NengoLMU `fixtures <https://docs.pytest.org/en/latest/fixture.html>`_
and `markers <https://docs.pytest.org/en/latest/example/markers.html>`_
to integrate well with existing tests.
See existing tests for examples, or consult

.. code-block:: bash

   pytest --pyargs lmu --fixtures

and

.. code-block:: bash

   pytest --pyargs lmu --markers

.. _pytest: https://docs.pytest.org/en/latest/

How to build the documentation
==============================

The documentation is built with Sphinx,
which should have been installed as part
of the :ref:`developer installation <dev-install>`.

After you've installed all the requirements,
run the following command from the root directory of ``lmu``
to build the documentation.
It will take a few minutes, as all examples are run
as part of the documentation building process.

.. code-block:: bash

   sphinx-build -vWaE docs docs/_build

If you wish to avoid running all examples, as some may
take some time depending on the hardware it is being 
run on, you can avoid these executions by running the 
following command.

.. code-block:: bash

   sphinx-build -vWaE docs docs/_build -D nbsphinx_execute=never

Depending on your environment,
you might have to set the Jupyter kernel
used to build the examples.
To set the kernel, use this command.

.. code-block:: bash

   sphinx-build -vW docs docs/_build -D nbsphinx_kernel_name=<kernelname>

.. _Pandoc: https://pandoc.org/

Getting help
============

If you have any questions about developing NengoLMU
or how you can best climb the learning curve
that NengoLMU and ``git`` present, please head to the
deep learning section on the `Nengo forum <https://
forum.nengo.ai/>`_ and we'll do our best to help you!
