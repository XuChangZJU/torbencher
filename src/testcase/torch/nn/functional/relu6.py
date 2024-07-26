import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.relu6)
class TorchNnFunctionalRelu6TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu6_correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with values between -10 and 10
        input_tensor = torch.rand(input_size) * 20 - 10
        # Apply the ReLU6 function
        result = torch.nn.functional.relu6(input_tensor)
        return result
