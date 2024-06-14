import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.clamp)
class TorchTensorClampTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clamp_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(1, 4)  
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1,5) 
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    # Generate a random tensor
    input_tensor = torch.randn(input_size)
    # Generate random min and max values for clamping
    min_val = random.uniform(-1.0, 1.0)
    max_val = random.uniform(1.0, 2.0)  # Ensure max_val > min_val
    # Apply clamp operation
    result = input_tensor.clamp(min_val, max_val)
    return result
