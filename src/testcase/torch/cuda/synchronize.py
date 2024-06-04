
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.synchronize)
class TorchCudaSynchronizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_synchronize_0(self):
        
        result = torch.cuda.synchronize()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_synchronize_1(self):
        
        a = 0
        result = torch.cuda.synchronize(a)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_synchronize_2(self):
        
        a = 0
        result = torch.cuda.synchronize(device=a)
        return result

