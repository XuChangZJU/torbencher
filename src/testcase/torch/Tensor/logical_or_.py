import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.logical_or_)
class TorchTensorLogicalUorUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_or__correctness(self):
        # Define the dimension and size of the tensors randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors
        tensor = torch.randn(input_size)
        other = torch.randn(input_size)

        # Perform the in-place logical OR operation
        tensor.logical_or_(other)

        # Return the modified tensor for assertion
        return tensor
