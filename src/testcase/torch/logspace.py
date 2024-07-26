import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.logspace)
class TorchLogspaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logspace_correctness(self):
        # Randomly generated start value for logspace
        start = random.uniform(-10.0, 10.0)
        # Randomly generated end value for logspace, ensuring it is not less than start for valid operation
        end = random.uniform(start, start + 20.0)
        # Random number of steps in the interval
        steps = random.randint(2, 10)

        result = torch.logspace(start, end, steps)
        return result
