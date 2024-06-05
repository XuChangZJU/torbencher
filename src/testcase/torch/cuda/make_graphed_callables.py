
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.make_graphed_callables)
class TorchCudaMakeGraphedCallablesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_make_graphed_callables_correctness(self):
        graph = torch.cuda.CUDAGraph()
        callables = []
        for _ in range(random.randint(1, 10)):
            callables.append(lambda x: x + 1)
        result = torch.cuda.make_graphed_callables(graph, callables)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_make_graphed_callables_large_scale(self):
        graph = torch.cuda.CUDAGraph()
        callables = []
        for _ in range(random.randint(1000, 10000)):
            callables.append(lambda x: x + 1)
        result = torch.cuda.make_graphed_callables(graph, callables)
        return result

