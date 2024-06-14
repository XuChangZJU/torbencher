import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.matmul)
class TorchMatmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_matmul_vector_vector(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        other = torch.randn(dim)
        result = torch.matmul(input, other)
        return result
    
    
    
    
    
    
    