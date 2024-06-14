import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Tanh)
class TorchNnTanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tanh_correctness(self):
    # Define the dimension and size of the input tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate a random tensor
    input_tensor = torch.randn(input_size)

    # Apply Tanh activation function
    tanh_activation = torch.nn.Tanh()
    output_tensor = tanh_activation(input_tensor)

    return output_tensor
