import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.inverse)
class TorchInverseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inverse_correctness(self):
    dim = random.randint(2, 4)  # Random dimension for the square matrix (must be 2 or higher)
    matrix_size = random.randint(2, 5)  # Random size for the square matrix dimension
    input_size = [matrix_size, matrix_size]
    
    tensor = torch.randn(input_size)
    
    # Make sure the tensor is invertible by adding a small identity matrix factor
    tensor = tensor + torch.eye(matrix_size) * 0.01
    
    result = torch.inverse(tensor)
    return result
