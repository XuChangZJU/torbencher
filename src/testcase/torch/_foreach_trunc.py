import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_trunc)
class TorchForeachtruncTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_trunc_correctness(self):
        # foreach_trunc requires the same number of elements for each tensor in the list
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        list_len = random.randint(1, 5)  # Random length for the list

        tensor_list = [torch.randn(input_size) for i in range(list_len)]
        result = torch._foreach_trunc(tensor_list)
        return result
