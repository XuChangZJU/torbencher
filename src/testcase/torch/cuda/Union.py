
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.Union)
class TorchCudaUnionTestCase(TorBencherTestCaseBase):
    def test_union_correctness(self):
        result = torch.cuda.Union
        return result

    def test_union_large_scale(self):
        result = torch.cuda.Union
        return result

