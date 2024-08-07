import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_log1p)
class TorchUforeachUlog1pTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_log1p_correctness(self):
        # foreach_log1p requires the input tensors to be in the same device
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensors = [torch.randn(input_size), torch.randn(input_size), torch.randn(input_size)]
        result = torch._foreach_log1p(input_tensors)
        return result
