import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.select_scatter)
class TorchSelectscatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_select_scatter_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim + 1)]
        index = random.randint(0, input_size[dim] - 1)  # Random index within the dimension

        input_tensor = torch.randn(input_size)
        src_size = input_size[:dim] + input_size[dim + 1:]
        src_tensor = torch.randn(src_size)

        result = torch.select_scatter(input_tensor, src_tensor, dim, index)
        return result
