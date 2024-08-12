import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.erfcx)
class TorchSpecialErfcxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_erfcx_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate a random tensor
        input_tensor = torch.randn(input_size)
        # Calculate the scaled complementary error function
        result = torch.special.erfcx(input_tensor)
        return result
