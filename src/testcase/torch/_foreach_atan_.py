import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_atan_)
class TorchUforeachUatanUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_atan_correctness(self):
        # foreach_atan_ operator applies on a list of tensor with the same size.
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        list_len = random.randint(1, 5)  # generate random length for the input list

        # Generate random list of tensors
        tensor_list = [torch.randn(input_size) for _ in range(list_len)]
        result = torch._foreach_atan_(tensor_list)
        return result
