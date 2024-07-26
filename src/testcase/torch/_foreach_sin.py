import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_sin)
class TorchForeachsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_sin_correctness(self):
        # foreach_sin requires the same size of tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        list_len = random.randint(1, 5)

        input_tensors = []
        for i in range(list_len):
            input_tensors.append(torch.randn(input_size))
        result = torch._foreach_sin(input_tensors)
        return result
