import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.use_deterministic_algorithms)
class TorchUseUdeterministicUalgorithmsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_use_deterministic_algorithms_correctness(self):
        # mode: whether to use deterministic algorithms
        mode = random.choice([True, False])
        # warn_only: whether to throw a warning instead of an error if a deterministic implementation is not available
        warn_only = random.choice([True, False])
        result = torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        return result
