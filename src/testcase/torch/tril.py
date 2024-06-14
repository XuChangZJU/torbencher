import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.tril)
class TorchTrilTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tril_correctness(self):
    dim = random.randint(2, 4)  # Random dimension for the tensors (at least 2 dimension is required)
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    input_tensor = torch.randn(input_size)
    diagonal = random.randint(-(input_size[0]-1), (input_size[1]-1)) # diagonal should be in range [-(input_size[0]-1), (input_size[1]-1)]
    result = torch.tril(input_tensor, diagonal)
    return result
