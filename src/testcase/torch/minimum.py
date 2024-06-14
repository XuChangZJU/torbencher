import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.minimum)
class TorchMinimumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_minimum_correctness(self):
        # Generate random dimension and size for input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors with the same size
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)
    
        # Calculate the element-wise minimum
        result = torch.minimum(input_tensor, other_tensor)
        return result
    