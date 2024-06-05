
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_default_dtype)
class TorchSetDefaultDtypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_default_dtype_correctness(self):
        dtype = torch.float32
        result = torch.set_default_dtype(dtype)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_default_dtype_large_scale(self):
        dtype = torch.float64
        result = torch.set_default_dtype(dtype)
        return result

