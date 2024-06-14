import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.vstack)
class TorchVstackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vstack_correctness(self):
        # Random number of tensors to stack
        num_tensors = random.randint(2, 5) 
        
        # Random small scale dimensions for each tensor
        tensors = []
        for _ in range(num_tensors):
            dim = random.randint(1, 3)
            size = [random.randint(1, 4) for _ in range(dim)]
            tensors.append(torch.randn(size))
        
        result = torch.vstack(tensors)
        return result
    
    
    
    