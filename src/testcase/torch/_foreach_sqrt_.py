import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_sqrt_)
class TorchForeachsqrtTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_sqrt_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        list_of_tensor = []
        num_of_tensor = random.randint(1, 5)  # Random number of tensors in the list
        for i in range(num_of_tensor):
            tensor = torch.randn(input_size)
            list_of_tensor.append(tensor)
        torch._foreach_sqrt_(list_of_tensor)
        return list_of_tensor
