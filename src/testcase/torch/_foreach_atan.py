import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_atan)
class TorchUforeachUatanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_atan_correctness(self):
        # foreach_atan requires the same number of elements for each Tensor in the list
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        list_len = random.randint(1, 5)  # Random length of the list

        tensor_list = [torch.randn(input_size) for i in range(list_len)]
        result = torch._foreach_atan(tensor_list)
        return result
