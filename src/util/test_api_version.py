import unittest
import packaging.version

import torch


def larger_than(version_str: str):
    version = packaging.version.parse(version_str)
    return unittest.skipIf(
        torch.__version__ <= (version.major, version.minor, version.micro),
        f"Skip this test because torch version is not larger than {version_str}",
    )


def less_than(version_str: str):
    version = packaging.version.parse(version_str)
    return unittest.skipIf(
        torch.__version__ >= (version.major, version.minor, version.micro),
        f"Skip this test because torch version is not less than {version_str}",
    )


def between(lowest_version_str: str, highest_version_str: str):
    lowest_version = packaging.version.parse(lowest_version_str)
    highest_version = packaging.version.parse(highest_version_str)
    return unittest.skipIf(
        torch.__version__
        >= (lowest_version.major, lowest_version.minor, lowest_version.micro)
        and torch.__version__
        <= (highest_version.major, highest_version.minor, highest_version.micro),
        f"Skip this test because torch version is not between {lowest_version_str} and {highest_version_str}",
    )


def equal(version_str: str):
    version = packaging.version.parse(version_str)
    return unittest.skipIf(
        torch.__version__ != (version.major, version.minor, version.micro),
        f"Skip this test because torch version is not equal to {version_str}",
    )
