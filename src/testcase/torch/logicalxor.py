import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logicalxor)
class TorchLogicalxorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_xor_correctness(self):
    # Define the dimension and size of the tensors
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random tensors with values 0 or 1
    input_tensor = torch.randint(0, 2, input_size) 
    other_tensor = torch.randint(0, 2, input_size)

    # Calculate the logical XOR of the tensors
    result = torch.logical_xor(input_tensor, other_tensor)
    
    # Return the result tensor
    return result
