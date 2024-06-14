import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cross)
class TorchTensorCrossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(3, 5)  # Random number of elements each dimension, must be at least 3 for cross product
        input_size = [num_of_elements_each_dim for _ in range(dim)] 
    
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        dim = random.randint(0, dim-1)  # Random dimension along which to compute the cross product
    
        result = torch.cross(tensor1, tensor2, dim)
        return result
    