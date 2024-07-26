import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.softshrink)
class TorchNnFunctionalSoftshrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softshrink_correctness(self):
        # Randomly generate dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate number of elements each dimension for the input tensor
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size list for the input tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Apply softshrink function
        result = torch.nn.functional.softshrink(input_tensor)
        return result
    
    
    
    