import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_cosh)
class TorchForeachcoshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_cosh_correctness(self):
        # foreach_cosh operator applies element-wise, so we test with a list of randomly sized tensors.
        num_tensors = random.randint(1, 5)
        tensor_list = []
        for i in range(num_tensors):
            dim = random.randint(1, 4)
            num_of_elements_each_dim = random.randint(1, 5)
            input_size = [num_of_elements_each_dim for i in range(dim)]
            tensor_list.append(torch.randn(input_size))  # No constraints on the input tensor values for torch.cosh
        result = torch._foreach_cosh(tensor_list)
        return result
