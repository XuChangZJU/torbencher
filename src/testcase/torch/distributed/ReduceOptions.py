
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.ReduceOptions)
class TorchReduceOptionsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reduce_options_correctness(self):
        result = torch.distributed.ReduceOptions.pybind11_type()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_reduce_options_large_scale(self):
        result = torch.distributed.ReduceOptions.pybind11_type()
        return result

