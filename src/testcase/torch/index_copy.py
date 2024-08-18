import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.index_copy)
class TorchIndexUcopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_copy_correctness(self):
        x = torch.zeros(5, 3)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        result = x.index_copy(0, index, t)
        return result
