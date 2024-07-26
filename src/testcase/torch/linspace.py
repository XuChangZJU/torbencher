import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linspace)
class TorchLinspaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linspace_correctness(self):
        # Randomly generate start and end values for the linspace
        start = random.uniform(-100, 100)
        end = random.uniform(-100, 100)

        # Ensure steps is a positive integer
        steps = random.randint(1, 100)

        result = torch.linspace(start, end, steps)
        return result
