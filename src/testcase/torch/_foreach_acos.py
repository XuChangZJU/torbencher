import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_acos)
class TorchUforeachUacosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_acos_correctness(self):
        # foreach_acos requires the input tensors to be in the range [-1, 1]
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.randn(input_size).clamp(min=-1.0, max=1.0) for _ in range(random.randint(1, 3))]
        result = torch._foreach_acos(tensor_list)
        return result
