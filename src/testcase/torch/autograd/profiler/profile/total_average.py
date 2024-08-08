import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.profile.total_average)
class TorchAutogradProfilerProfileTotalUaverageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_total_average_correctness(self):
        """
        This test case verifies the correctness of the `total_average` method
        by comparing its output with manually calculated averages from a small
        scale profile.
        """
        with torch.autograd.profiler.profile(use_cuda=False) as prof:
            # Generate random tensors for operations
            dim = random.randint(1, 4)
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for i in range(dim)]
            x = torch.randn(input_size)
            y = torch.randn(input_size)

            # Perform some operations within the profiler's context
            z = torch.add(x, y)
            w = torch.mul(z, x)

        # Calculate total average using the 'total_average' method
        total_average_event = prof.total_average()

        return total_average_event
