import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sym_max)
class TorchSymmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sym_max_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate two random tensors with the same shape
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)

        # Calculate maximum between corresponding elements of tensor1 and tensor2
        result = torch.max(tensor1, tensor2)
        return result
