import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_abs_)
class TorchForeachabsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_abs_correctness(self):
        # foreach_abs_ operator applies element-wise abs to each tensor in a list.
        # The following code generates a list of random tensors and apply foreach_abs_ to it.
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor_list = [torch.randn(input_size) for _ in range(random.randint(1, 3))]
        result = torch._foreach_abs_(tensor_list)
        return result
