import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.logical_and)
class TorchLogicalUandTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logical_and_correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors with values that can be interpreted as True or False
        input_tensor = torch.randint(0, 2,
                                     input_size).bool()  # Generates random tensor with 0 or 1, representing False or True
        other_tensor = torch.randint(0, 2,
                                     input_size).bool()  # Generates random tensor with 0 or 1, representing False or True

        # Calculate the logical AND of the tensors
        result = torch.logical_and(input_tensor, other_tensor)

        # Return the result tensor
        return result
