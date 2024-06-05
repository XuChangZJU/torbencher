
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.get_num_interop_threads)
class TorchGetNumInteropThreadsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_num_interop_threads_correctness(self):
        result = torch.get_num_interop_threads()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_num_interop_threads_large_scale(self):
        result = torch.get_num_interop_threads()
        return result

