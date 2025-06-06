# Automatically generated by nengo-bones, do not edit this file directly

import io
import pathlib
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = pathlib.Path(__file__).parent
version = runpy.run_path(str(root / "keras_lmu" / "version.py"))["version"]

install_req = [
    "packaging>=20.9",
    "scipy>=1.0.0",
    "tensorflow>=2.8.0",
]
docs_req = [
    "matplotlib>=3.8.4",
    "jupyter>=1.0.0",
    "seaborn>=0.9.0",
    "sphinx>=1.8",
    "nbsphinx>=0.3.5",
    "nengo_sphinx_theme>=1.2.0",
    "numpydoc>=0.6",
]
optional_req = []
tests_req = [
    "pytest>=6.1.0",
    "pytest-rng>=1.0.0",
]

setup(
    name="keras-lmu",
    version=version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://www.nengo.ai/keras-lmu",
    include_package_data=False,
    license="Free for non-commercial use",
    description="Keras implementation of Legendre Memory Units",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": docs_req + optional_req + tests_req,
        "docs": docs_req,
        "optional": optional_req,
        "tests": tests_req,
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
