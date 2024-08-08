import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_lgamma_)
class TorchUforeachUlgammaUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_lgamma_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.randn(input_size) for _ in
                       range(random.randint(1, 3))]  # Generate a list of random tensors
        torch._foreach_lgamma_(tensor_list)  # Apply _foreach_lgamma_ in-place
        return tensor_list[0]  # Return one of the modified tensors to show the effect
