import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_ceil)
class TorchUforeachUceilTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_ceil_correctness(self):
        # foreach_ceil requires the input to be a List[Tensor]
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor_list = [torch.randn(input_size) for _ in range(random.randint(1, 3))]
        result = torch._foreach_ceil(tensor_list)
        return result
