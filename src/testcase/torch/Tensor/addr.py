import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.addr)
class TorchTensorAddrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addr_correctness(self):
    # Random size for the vectors
    vec1_size = random.randint(1, 10)
    vec2_size = random.randint(1, 10)
    
    # Randomly generated vectors
    vec1 = torch.randn(vec1_size)
    vec2 = torch.randn(vec2_size)
    
    # Randomly generated matrix to apply addr operation
    mat = torch.randn(vec1_size, vec2_size)
    
    # Perform addr operation
    result = mat.addr(vec1, vec2)
    return result
