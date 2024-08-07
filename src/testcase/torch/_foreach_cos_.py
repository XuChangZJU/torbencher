import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_cos_)
class TorchUforeachUcosUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_cos_correctness(self):
        # foreach_cos_ operator applies to a list of tensors
        # generate random number of tensors in the list
        num_tensors = random.randint(1, 5)
        # generate random dimension for the tensors
        dim = random.randint(1, 4)
        # generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # generate input size for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # generate a list of random tensors
        tensor_list = [torch.randn(input_size) for _ in range(num_tensors)]
        # apply foreach_cos_ operator
        torch._foreach_cos_(tensor_list)
        # return the first tensor in the list after applying foreach_cos_
