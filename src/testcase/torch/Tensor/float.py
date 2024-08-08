import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.float)
class TorchTensorFloatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_float_correctness(self):
        # Define the dimension and size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor of data type other than float32
        tensor = torch.randint(0, 100, input_size).double()

        # Call the float() method
        result = tensor.float()

        return result
