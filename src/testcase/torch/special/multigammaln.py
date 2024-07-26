import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.multigammaln)
class TorchSpecialMultigammalnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multigammaln_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor with elements greater than (p - 1) / 2
        input_tensor = torch.rand(input_size) + (random.randint(2, 5) - 1) / 2
        # Random integer p
        p = random.randint(1, 5)
        # Calculate the result of torch.special.multigammaln
        result = torch.special.multigammaln(input_tensor, p)
        return result
    