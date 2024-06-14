import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.slogdet)
class TorchTensorSlogdetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_slogdet_correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the square matrix (must be at least 2x2)
        size = random.randint(2, 5)  # Random size for each dimension (must be at least 2)
        input_size = [size, size]  # Square matrix
    
        tensor = torch.randn(input_size)
        sign, logabsdet = tensor.slogdet()
        return sign, logabsdet
    