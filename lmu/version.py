"""LMU version information.

We use semantic versioning (see http://semver.org/).
and conform to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""

name = "lmu"
version_info = (0, 2, 0)  # (major, minor, patch)
dev = None

version = (
    f"{'.'.join(str(v) for v in version_info)}{f'.dev{dev}' if dev is not None else ''}"
)
