import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.isreal)
class TorchIsrealTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isreal_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    input_tensor = torch.randn(input_size)  # Generate a random tensor with real values
    input_tensor[0] = input_tensor[0] + input_tensor[0] * 1j # Manually set the first element to be complex
    result = torch.isreal(input_tensor)
    return result
