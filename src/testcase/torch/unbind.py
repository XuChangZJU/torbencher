import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.unbind)
class TorchUnbindTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unbind_correctness(self):
        dim = random.randint(0, 3)  # Random dimension to remove, should be valid dimension index.
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in
                      range(dim + 1)]  # dim + 1 to make sure the dimension to be removed exists.

        input_tensor = torch.randn(input_size)
        result = torch.unbind(input_tensor, dim)
        return result
