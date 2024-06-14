import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.outer)
class TorchTensorOuterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_outer_correctness(self):
    # Random size for the vectors
    size1 = random.randint(1, 10)
    size2 = random.randint(1, 10)
    
    # Generate random vectors
    vec1 = torch.randn(size1)
    vec2 = torch.randn(size2)
    
    # Compute the outer product
    result = vec1.outer(vec2)
    return result
