import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.cumprod_)
class TorchTensorCumprodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cumprod__correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random tensor
        input_tensor = torch.randn(input_size)
        # Random dimension to calculate cumulative product
        dim = random.randint(0, len(input_size) - 1)
        # Calculate cumulative product in-place
        input_tensor.cumprod_(dim)
        # Return the tensor after in-place operation
        return input_tensor
    
    
    
    