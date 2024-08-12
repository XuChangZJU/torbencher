import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.EnforceUnique)
class TorchAutogradProfilerEnforceuniqueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_enforce_unique_correctness(self):
        # Generate a random number of keys
        num_keys = random.randint(1, 10)

        # Test with unique keys (should not raise an error)
        for i in range(num_keys):
            torch.autograd.profiler.EnforceUnique().see(i)

        # Test with duplicate keys (should raise an error)
        try:
            torch.autograd.profiler.EnforceUnique().see(0)
        except RuntimeError as e:
            f"Expected error for duplicate keys: {e}"
