# pylint: disable=missing-docstring

import subprocess
import sys

# check if GPU support is available
# note: we run this in a subprocess because list_physical_devices()
# will fix certain process-level TensorFlow configuration
# options the first time it is called
tf_gpu_installed = not subprocess.call(
    [
        sys.executable,
        "-c",
        "import sys; "
        "import tensorflow as tf; "
        "sys.exit(len(tf.config.list_physical_devices('GPU')) == 0)",
    ]
)
