import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_lgamma)
class TorchForeachlgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_lgamma_correctness(self):
        # foreach_lgamma requires the input to be a list of tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        num_tensors = random.randint(1, 3)  # Random number of tensors in the list
        tensor_list = [torch.randn(input_size) for _ in range(num_tensors)]
        result = torch._foreach_lgamma(tensor_list)
        return result
