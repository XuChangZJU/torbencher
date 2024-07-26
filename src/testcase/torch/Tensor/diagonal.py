import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.diagonal)
class TorchTensorDiagonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diagonal_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(3, 5)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random size for the input tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified size
        input_tensor = torch.randn(input_size)
        # Generate random offset
        offset = random.randint(-num_of_elements_each_dim + 1, num_of_elements_each_dim - 1)
        # Generate random dim1 and dim2, making sure dim1 != dim2
        dim1 = random.randint(0, dim - 1)
        dim2 = random.randint(0, dim - 1)
        while dim1 == dim2:
            dim2 = random.randint(0, dim - 1)
        result = input_tensor.diagonal(offset, dim1, dim2)
        return result
