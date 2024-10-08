import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_cos)
class TorchUforeachUcosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_cos_correctness(self):
        # foreach_cos input is a List[Tensor]
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        num_of_tensor = random.randint(1, 5)  # Random number of tensors in the list
        input_tensors = [torch.randn(input_size) for i in range(num_of_tensor)]
        result = torch._foreach_cos(input_tensors)
        return result
