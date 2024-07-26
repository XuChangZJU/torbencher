import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.dim_order)
class TorchTensorDimorderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dim_order_correctness(self):
        # Randomly choose dimensions for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = tuple(num_of_elements_each_dim for _ in range(dim))
    
        # Create a random tensor with the chosen dimensions
        tensor = torch.randn(input_size)
        result = tensor.dim()
        return result
    
    def test_dim_order_channels_last(self):
        # Randomly choose dimensions for the tensor
        dim = 4  # Ensure 4 dimensions for channels_last format
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = tuple(num_of_elements_each_dim for _ in range(dim))
    
        # Create a random tensor with the chosen dimensions and channels_last memory format
        tensor = torch.randn(input_size).to(memory_format=torch.channels_last)
        result = tensor.dim()
        return result
    
    print()
    print()
    
    