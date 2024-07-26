import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.logical_xor_)
class TorchTensorLogicalxorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_xor__correctness(self):
        # Define the dimension of the tensors
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensors
        input_tensor = torch.randint(0, 2, input_size).float()  # Generate random tensor with 0 or 1
        other_tensor = torch.randint(0, 2, input_size).float()  # Generate random tensor with 0 or 1
        # Perform the in-place logical XOR operation
        input_tensor.logical_xor_(other_tensor)
        # Return the result tensor
        return input_tensor
