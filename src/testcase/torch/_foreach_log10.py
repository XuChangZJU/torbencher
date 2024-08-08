import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_log10)
class TorchUforeachUlog10TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_log10_correctness(self):
        # foreach_log10 requires the input to be a list of tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        list_len = random.randint(1, 5)

        tensor_list = [torch.randn(input_size) for i in range(list_len)]
        result = torch._foreach_log10(tensor_list)
        return result
