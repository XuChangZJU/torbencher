import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ELU)
class TorchNnEluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_elu_correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Apply ELU activation
        elu_op = torch.nn.ELU()
        output_tensor = elu_op(input_tensor)

        return output_tensor
