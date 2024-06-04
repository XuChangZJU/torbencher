
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_summary)
class TorchCudaMemorySummaryTestCase(TorBencherTestCaseBase):
    def test_memory_summary_0(self):
        
        a = torch.device('cuda')
        result = torch.cuda.memory_summary(a)
        return result
    def test_memory_summary_1(self):
        
        a = torch.device('cuda')
        result = torch.cuda.memory_summary(device=a)
        return result
    def test_memory_summary_2(self):
        
        a = torch.device('cuda')
        b = True
        result = torch.cuda.memory_summary(a, b)
        return result
    def test_memory_summary_3(self):
        
        a = torch.device('cuda')
        b = True
        result = torch.cuda.memory_summary(device=a, abbreviated=b)
        return result

