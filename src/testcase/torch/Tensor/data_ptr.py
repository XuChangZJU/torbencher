import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.data_ptr)
class TorchTensorDataUptrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_data_ptr_correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor
        tensor = torch.randn(input_size)

        # Get the data pointer of the tensor
        data_ptr = tensor.data_ptr()

        return data_ptr
