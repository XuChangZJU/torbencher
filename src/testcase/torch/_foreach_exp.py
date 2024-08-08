import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_exp)
class TorchUforeachUexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_exp_correctness(self):
        # foreach_exp requires the length of input list larger than 0
        list_len = random.randint(1, 5)
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensors = [torch.randn(input_size) for _ in range(list_len)]
        result = torch._foreach_exp(input_tensors)
        return result
