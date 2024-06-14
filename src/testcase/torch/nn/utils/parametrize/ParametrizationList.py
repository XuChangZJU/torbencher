import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.parametrize.ParametrizationList)
class TorchNnUtilsParametrizeParametrizationlistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_parametrization_list_correctness(self):
    class MyParametrization(nn.Module):
        def forward(self, X):
            return X * 2
