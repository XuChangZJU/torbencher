import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_sigmoid)
class TorchUforeachUsigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_foreach_sigmoid_correctness(self):
        # foreach_sigmoid requires the length of input list to be larger than 0
        input_list_length = random.randint(1, 5)
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size based on dim and num_of_elements_each_dim
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a list of random tensors
        tensor_list = [torch.randn(input_size) for i in range(input_list_length)]
        result = torch._foreach_sigmoid(tensor_list)
        return result
