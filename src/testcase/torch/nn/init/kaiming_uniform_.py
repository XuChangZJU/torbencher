import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.init.kaiming_uniform_)
class TorchNnInitKaimingUuniformUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_kaiming_uniform_correctness(self):
        # Randomly generate dimensions for the tensor
        dim = random.randint(2, 4)  # Ensure at least 2 dimensions
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor with the generated dimensions
        tensor = torch.empty(input_size)

        # Randomly generate the negative slope for leaky_relu
        a = random.uniform(0.01, 0.5)

        # Randomly choose mode between 'fan_in' and 'fan_out'
        mode = random.choice(['fan_in', 'fan_out'])

        # Randomly choose nonlinearity between 'relu' and 'leaky_relu'
        nonlinearity = random.choice(['relu', 'leaky_relu'])

        # Apply kaiming_uniform_ initialization
        result = torch.nn.init.kaiming_uniform_(tensor, a, mode, nonlinearity)
        return result
