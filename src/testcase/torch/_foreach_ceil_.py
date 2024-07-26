import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_ceil_)
class TorchForeachceilTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_ceil_correctness(self):
        # foreach_ceil_ is an inplace function. So we test its correctness by comparing the result before and after the function call.
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.randn(input_size) for _ in range(random.randint(1, 5))]
        expected_result = [torch.ceil(tensor) for tensor in tensor_list]
        torch._foreach_ceil_(tensor_list)
        return tensor_list
