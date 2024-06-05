
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.Tuple)
class TorchCudaTupleTestCase(TorBencherTestCaseBase):
    def test_tuple_correctness(self):
        result = torch.cuda.Tuple
        return result

    def test_tuple_large_scale(self):
        result = torch.cuda.Tuple
        return result

