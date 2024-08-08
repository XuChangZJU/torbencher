import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_neg)
class TorchUforeachUnegTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_neg_correctness(self):
        # foreach_neg requires the same dtype
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor_list = [torch.randn(input_size), torch.randn(input_size), torch.randn(input_size)]
        result = torch._foreach_neg(tensor_list)
        return result
