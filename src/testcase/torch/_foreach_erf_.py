import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_erf_)
class TorchUforeachUerfUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_erf_correctness(self):
        # foreach_erf_ requires the same number of elements for each tensor in the list
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        list_len = random.randint(1, 5)
        input_tensors = [torch.randn(input_size) for _ in range(list_len)]
        torch._foreach_erf_(input_tensors)
        return input_tensors
