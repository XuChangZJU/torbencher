import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addr)
class TorchAddrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addr_correctness(self):
        # Randomly generate input tensor, vec1, and vec2 with compatible dimensions
        m = random.randint(1, 5)  
        n = random.randint(1, 5)  
        input_tensor = torch.randn(m, n)
        vec1 = torch.randn(m)
        vec2 = torch.randn(n)
        result = torch.addr(input_tensor, vec1, vec2)
        return result
    
    
    
    
    
    
    