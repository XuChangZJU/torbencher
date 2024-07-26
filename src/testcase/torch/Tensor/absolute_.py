import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.absolute_)
class TorchTensorAbsoluteTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_absolute__correctness(self):
        # Randomly generate the dimension of the input tensor.
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension.
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor.
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor.
        input_tensor = torch.randn(input_size)
        # Apply the absolute_ function.
        input_tensor.absolute_()
        # Return the tensor after applying absolute_.
        return input_tensor
