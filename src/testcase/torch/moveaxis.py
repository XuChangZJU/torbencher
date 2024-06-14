import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.moveaxis)
class TorchMoveaxisTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_moveaxis_correctness_small_scale(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    input_tensor = torch.randn(input_size)
    source = random.randint(0, len(input_size) - 1) # source should be an integer in the range [0, dim - 1]
    destination = random.randint(0, len(input_size) - 1) # destination should be an integer in the range [0, dim - 1]
    result = torch.moveaxis(input_tensor, source, destination)
    return result
