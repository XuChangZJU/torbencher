import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_contiguous)
class TorchTensorIscontiguousTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_contiguous_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size)
        result = tensor.is_contiguous()
        return result

    def test_is_contiguous_non_contiguous(self):
        dim = random.randint(2, 4)  # Random dimension for the tensor, at least 2 to ensure non-contiguous possibility
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor = torch.randn(input_size).transpose(0, 1)  # Transpose to make it non-contiguous
        result = tensor.is_contiguous()
        return result
