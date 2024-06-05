
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_arch_list)
class TorchCudaGetArchListTestCase(TorBencherTestCaseBase):
    def test_get_arch_list_correctness(self):
        result = torch.cuda.get_arch_list()
        return result

    def test_get_arch_list_large_scale(self):
        result = torch.cuda.get_arch_list()
        return result

