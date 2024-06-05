
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.CUDAGraph)
class TorchCudaCUDAGraphTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_cudagraph_correctness(self):
        graph = torch.cuda.CUDAGraph()
        result = graph.capture_begin()
        return result

    @test_api_version.larger_than("1.7.0")
    def test_cudagraph_large_scale(self):
        graph = torch.cuda.CUDAGraph()
        result = graph.capture_begin()
        return result

