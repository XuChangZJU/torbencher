import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.transpose)
class TorchTensorTransposeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transpose_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random size list for tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        input_tensor = torch.randn(input_size)
        # Generate random dim0 and dim1, make sure they are valid
        dim0 = random.randint(0, dim - 1)
        dim1 = random.randint(0, dim - 1)
        # Transpose the tensor
        result = input_tensor.transpose(dim0, dim1)
        return result
