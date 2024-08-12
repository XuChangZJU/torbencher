import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_log2_)
class TorchUforeachUlog2UTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_log2_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.abs(torch.randn(input_size)) + 1e-6 for _ in range(random.randint(1, 3))]
        torch._foreach_log2_(tensor_list)
        return tensor_list
