import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.resize_)
class TorchTensorResizeUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_resize_correctness(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        return x.resize_(2, 2)
