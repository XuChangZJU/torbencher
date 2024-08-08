import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.diag)
class TorchTensorDiagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_Tensor_diag_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        input_tensor = torch.randn(input_size)
        # Generate random diagonal
        diagonal = random.randint(-max(input_size), max(input_size) - 1)  # diagonal must be in range [-n+1, n-1]
        # Calculate the result of diag
        result = input_tensor.diag(diagonal)
        # Return the result
        return result
