import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.softmax)
class TorchSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmax_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        dim_to_apply_softmax = random.randint(0, dim - 1)  # Randomly select dimension to apply softmax, must be within tensor dimensions
    
        result = torch.softmax(input_tensor, dim_to_apply_softmax)
        return result
    