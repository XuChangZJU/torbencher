import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.dim)
class TorchTensorDimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_Tensor_dim_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(0, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        tensor = torch.randn(input_size)

        # Get the number of dimensions of the tensor
        result = tensor.dim()
        return result
