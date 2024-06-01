"""add.py"""

import torch
import src.util.test_api_version as test_api_version
from ..TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util.decorator import test_api


@test_api(torch.add)
class TorchAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_4d(self, input=None):
        if input is not None:
            result = torch.add(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.add(a, b, alpha=10)
        return [result, [a, b, 10]]
