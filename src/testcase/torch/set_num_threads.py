
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_num_threads)
class TorchSetNumThreadsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_num_threads_correctness(self):
        num_threads = random.randint(1, 10)
        result = torch.set_num_threads(num_threads)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_num_threads_large_scale(self):
        num_threads = random.randint(1000, 10000)
        result = torch.set_num_threads(num_threads)
        return result

