import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_trunc_)
class TorchUforeachUtruncUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_trunc_correctness(self):
        # foreach_trunc_ operates on a list of tensors.
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        num_of_tensors = random.randint(1, 3)  # Random number of tensors in the list.
        tensor_list = [torch.randn(input_size) for i in range(num_of_tensors)]
        result = torch._foreach_trunc_(tensor_list)  # Applies trunc in place.
        return tensor_list  # Return the modified list to check for in-place modification.
