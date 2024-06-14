import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.amin)
class TorchAminTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_amin_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for i in range(dim)]
    input_tensor = torch.randn(input_size)  # Random tensor
    dim_to_reduce = random.randint(0, dim - 1)  # Random valid dimension to reduce
    result = torch.amin(input_tensor, dim_to_reduce)
    return result
