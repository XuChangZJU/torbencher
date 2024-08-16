import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nanmean)
class TorchNanmeanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_nanmean_correctness(self):
        x = torch.tensor([[torch.nan, 1, 2], [1, 2, 3]])
        return x.nanmean(), x.nanmean(dim=0), torch.tensor([torch.nan]).nanmean()
