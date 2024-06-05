
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.List)
class TorchCudaListTestCase(TorBencherTestCaseBase):
    def test_list_correctness(self):
        result = torch.cuda.List
        return result

    def test_list_large_scale(self):
        result = torch.cuda.List
        return result

