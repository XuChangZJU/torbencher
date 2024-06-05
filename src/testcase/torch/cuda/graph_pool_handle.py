
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.graph_pool_handle)
class TorchCudaGraphPoolHandleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_graph_pool_handle_correctness(self):
        result = torch.cuda.graph_pool_handle()
        return result

    @test_api_version.larger_than("1.7.0")
    def test_graph_pool_handle_large_scale(self):
        result = torch.cuda.graph_pool_handle()
        return result

