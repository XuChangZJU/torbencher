import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.no_grad)
class TorchNoUgradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_no_grad_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor1 = torch.randn(input_size, requires_grad=True)
        tensor2 = torch.randn(input_size, requires_grad=True)
        alpha = random.uniform(0.1, 10.0)  # Random alpha value between 0.1 and 10.0

        with torch.no_grad():
            result = torch.add(tensor1, tensor2, alpha=alpha)

        return result
