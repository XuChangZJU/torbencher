
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_summary)
class TorchCudaMemorySummaryTestCase(TorBencherTestCaseBase):
    def test_memory_summary_0(self, input=None):
        if input is not None:
            result = torch.cuda.memory_summary(input[0])
            return [result, input]
        a = torch.device('cuda')
        result = torch.cuda.memory_summary(a)
        return [result, [a]]
    def test_memory_summary_1(self, input=None):
        if input is not None:
            result = torch.cuda.memory_summary(device=input[0])
            return [result, input]
        a = torch.device('cuda')
        result = torch.cuda.memory_summary(device=a)
        return [result, [a]]
    def test_memory_summary_2(self, input=None):
        if input is not None:
            result = torch.cuda.memory_summary(input[0], input[1])
            return [result, input]
        a = torch.device('cuda')
        b = True
        result = torch.cuda.memory_summary(a, b)
        return [result, [a, b]]
    def test_memory_summary_3(self, input=None):
        if input is not None:
            result = torch.cuda.memory_summary(device=input[0], abbreviated=input[1])
            return [result, input]
        a = torch.device('cuda')
        b = True
        result = torch.cuda.memory_summary(device=a, abbreviated=b)
        return [result, [a, b]]

