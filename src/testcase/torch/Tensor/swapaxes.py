import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.swapaxes)
class TorchTensorSwapaxesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_swapaxes_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(2, 4)  # Dimension should be at least 2 for swapaxes
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensor
        input_tensor = torch.randn(input_size)
    
        # Generate random axes to swap, ensuring they are within the valid range
        axis0 = random.randint(0, dim - 1)
        axis1 = random.randint(0, dim - 1)
    
        # Perform swapaxes operation
        result = input_tensor.swapaxes(axis0, axis1)
        return result
    
    
    
    