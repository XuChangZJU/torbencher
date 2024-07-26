import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch._foreach_frac_)
class TorchForeachfracTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_frac_correctness(self):
        num_of_tensors = random.randint(1, 4)  # Random number of tensors in the list
        dim = random.randint(1, 4)  # Random dimension for each tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension

        tensors = [torch.randn([num_of_elements_each_dim for _ in range(dim)]) for _ in range(num_of_tensors)]

        torch._foreach_frac_(tensors)
        return tensors
