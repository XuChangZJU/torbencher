import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.real)
class TorchTensorRealTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_Tensor_real_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random complex tensor
        complex_tensor = torch.randn(input_size, dtype=torch.cfloat)
        # Get real values of the complex tensor
        real_tensor = complex_tensor.real
        return real_tensor
