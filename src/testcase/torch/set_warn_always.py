import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.set_warn_always)
class TorchSetwarnalwaysTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_warn_always_correctness(self):
        # Generate random boolean value for b
        b = random.choice([True, False])

        # Call torch.set_warn_always with the random boolean value
        result = torch.set_warn_always(b)
        return result
