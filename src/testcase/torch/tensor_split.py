import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.tensor_split)
class TorchTensorUsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tensor_split_correctness(self):
        x = torch.arange(14).reshape(2, 7)
        return torch.tensor_split(x, (1, 6), dim=1)
