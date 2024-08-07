import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.leaky_relu)
class TorchNnFunctionalLeakyUreluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_leaky_relu_correctness(self):
        # Randomly generate the input tensor's dimension and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)  # Generate a random tensor with the specified size
        result = torch.nn.functional.leaky_relu(input_tensor)
        return result
