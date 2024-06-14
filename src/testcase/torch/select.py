import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.select)
class TorchSelectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_select_correctness(self):
    dim = random.randint(0, 3)  # Random dimension to slice
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for i in range(random.randint(1, 4))]  # Random input size
    index = random.randint(0, input_size[dim] - 1)  # Random index within the dimension's bounds
    input_tensor = torch.randn(input_size)
    result = torch.select(input_tensor, dim, index)
    return result
