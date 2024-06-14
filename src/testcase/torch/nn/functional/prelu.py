import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.prelu)
class TorchNnFunctionalPreluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_prelu_correctness(self):
    # Random dimension for the input tensor
    dim = random.randint(1, 4)
    # Random number of elements for each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for _ in range(dim)]

    # Generate random input tensor
    input_tensor = torch.randn(input_size)

    # Generate random weight tensor
    if dim >= 2:
        # If input tensor has more than 1 dimension, weight size should match the number of input channels
        weight_size = input_tensor.size(1)
        weight = torch.randn(weight_size)
    else:
        # If input tensor has 1 dimension, weight should be a scalar
        weight = torch.randn(1)

    # Apply PReLU function
    result = torch.nn.functional.prelu(input_tensor, weight)
    return result
