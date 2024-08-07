import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_neg_)
class TorchUforeachUnegUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_neg_correctness(self):
        # foreach_neg_ operator applies on a list of tensors.
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        list_len = random.randint(1, 5)

        # Generate a list of random tensors.
        tensor_list = [torch.randn(input_size) for i in range(list_len)]
        result = torch._foreach_neg_(tensor_list)
        return result
