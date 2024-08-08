import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.numpy)
class TorchTensorNumpyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_numpy_correctness(self):
        """
        Test the correctness of torch.Tensor.numpy() with small scale random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor on CPU
        tensor = torch.randn(input_size)

        # Convert the tensor to a NumPy array
        numpy_array = tensor.numpy()

        return numpy_array
