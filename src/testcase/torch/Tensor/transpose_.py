import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.transpose_)
class TorchTensorTransposeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transpose__correctness(self):
        dim = random.randint(2, 4)  # Random dimension for the tensors, should be larger than 1
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size)
        dim0 = random.randint(0, dim - 1)  # Random dim0
        dim1 = random.randint(0, dim - 1)  # Random dim1
        result = tensor.transpose_(dim0, dim1)
        return result