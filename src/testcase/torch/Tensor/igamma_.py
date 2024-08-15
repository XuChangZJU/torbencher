import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.igamma_)
class TorchTensorIgammaUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_igamma__correctness(self):
        a1 = torch.tensor([5.0, 4.0, 3.0])
        a2 = torch.tensor([3.0, 4.0, 5.0])
        a1.igamma_(a2)
        return a1
