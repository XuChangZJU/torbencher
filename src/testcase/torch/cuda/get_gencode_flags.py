
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.get_gencode_flags)
class TorchCudaGetGencodeFlagsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.5.0")
    def test_get_gencode_flags_correctness(self):
        result = torch.cuda.get_gencode_flags()
        return result

    @test_api_version.larger_than("1.5.0")
    def test_get_gencode_flags_large_scale(self):
        result = torch.cuda.get_gencode_flags()
        return result

