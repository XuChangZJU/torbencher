import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version


class TorchAsUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_as_tensor_correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create random NumPy array
        cpu = torch.device("cpu")
        data_numpy = torch.randn(input_size).to(cpu).numpy()

        # Convert NumPy array to tensor using torch.as_tensor
        tensor_from_numpy = torch.as_tensor(data_numpy)

        # Modify a value in the tensor
        tensor_from_numpy[0] = -1

        # Return the modified tensor
        return tensor_from_numpy
