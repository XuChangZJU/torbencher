import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.narrow)
class TorchNarrowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_narrow_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    input_tensor = torch.randn(input_size)
    dim = random.randint(0, len(input_size) - 1)  # Random valid dimension
    start = random.randint(0, input_size[dim] - 1)  # Random valid start index
    length = random.randint(1, input_size[dim] - start)  # Random valid length
    result = torch.narrow(input_tensor, dim, start, length)
    return result
