import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.take)
class TorchTakeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_take_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for i in range(dim)]
    input_tensor = torch.randn(input_size)
    
    # Generate valid indices within the range of the input tensor
    number_of_indices = random.randint(1, input_tensor.numel())
    indices = torch.randint(0, input_tensor.numel(), (number_of_indices,), dtype=torch.long)
    
    result = torch.take(input_tensor, indices)
    return result
