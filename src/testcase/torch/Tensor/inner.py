import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.inner)
class TorchTensorInnerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inner_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the first tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension for the first tensor

        # Ensure the last dimension of tensor1 matches the first dimension of tensor2
        tensor1_size = [num_of_elements_each_dim for _ in range(dim)]
        tensor2_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor1 = torch.randn(tensor1_size)
        tensor2 = torch.randn(tensor2_size)

        result = tensor1.inner(tensor2)
        return result
