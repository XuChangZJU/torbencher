import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.parametrize.removeparametrizations)
class TorchNnUtilsParametrizeRemoveparametrizationsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
        def __init__(self, input_size):
        super(TestModule, self).__init__()
        self.param = nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        return x * self.param
