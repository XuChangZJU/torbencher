import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.sign)
class TorchTensorSignTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sign_correctness(self):
        """
        Test the correctness of torch.Tensor.sign() with small scale random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor with values ranging from -10.0 to 10.0
        input_tensor = torch.rand(input_size) * 20.0 - 10.0
        result = input_tensor.sign()
        return result
