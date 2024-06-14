import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwiseor)
class TorchBitwiseorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_or_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input = torch.randint(0, 10, input_size)  # Generate random tensor of integers between 0 and 9
        other = torch.randint(0, 10, input_size)  # Generate random tensor of integers between 0 and 9
        result = torch.bitwise_or(input, other)
        return result
    