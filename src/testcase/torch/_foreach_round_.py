import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_round_)
class TorchUforeachUroundUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_round_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.randn(input_size) for _ in
                       range(random.randint(1, 3))]  # Generate a list of random tensors
        torch._foreach_round_(tensor_list)  # Apply _foreach_round_ to the list of tensors
        return tensor_list[0]  # Return the first tensor to check the effect of _foreach_round_
