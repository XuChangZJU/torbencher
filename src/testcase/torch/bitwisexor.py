import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwisexor)
class TorchBitwisexorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_xor_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input = torch.randint(0, 10, input_size) # generate random tensor with dtype=torch.int64
        other = torch.randint(0, 10, input_size) # generate random tensor with dtype=torch.int64
        result = torch.bitwise_xor(input, other)
        return result
    