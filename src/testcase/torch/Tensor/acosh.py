import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.acosh)
class TorchTensorAcoshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_acosh__correctness(self):
    # Randomly generate the dimension of the input tensor
    dim = random.randint(1, 4)
    # Randomly generate the number of elements in each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Create a list representing the size of the input tensor
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Generate a random tensor with values greater than 1
    input_tensor = torch.randn(input_size).abs() + 1
    # Apply the acosh_ operation in-place
    input_tensor.acosh_()
    # Return the tensor after applying acosh_
    return input_tensor
