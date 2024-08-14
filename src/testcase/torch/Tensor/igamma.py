import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.igamma)
class TorchTensorIgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_igamma_correctness(self):
        a1 = torch.tensor([4.0])
        a2 = torch.tensor([3.0, 4.0, 5.0])
        return a1.igamma(a2)
