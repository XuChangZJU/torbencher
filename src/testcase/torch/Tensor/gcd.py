import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.gcd)
class TorchTensorGcdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_gcd_correctness(self):
        # Generate random dimensions for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors with integer values
        tensor1 = torch.randint(low=-10, high=10, size=input_size)
        tensor2 = torch.randint(low=-10, high=10, size=input_size)

        # Calculate the GCD
        result = tensor1.gcd(tensor2)

        return result
