import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PReLU)
class TorchNnPreluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_prelu_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size based on dimensions

    input_tensor = torch.randn(input_size)  # Generate random input tensor
    num_parameters = random.choice([1, input_tensor.size(1) if len(input_tensor.size()) > 1 else 1])  # Choose valid num_parameters
    init_value = random.uniform(0.1, 10.0)  # Random initial value for 'a'

    prelu = torch.nn.PReLU(num_parameters, init_value)  # Initialize PReLU with random parameters
    result = prelu(input_tensor)  # Apply PReLU to the input tensor
    return result
