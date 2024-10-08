import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.log2_)
class TorchTensorLog2UTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_log2__correctness(self):
        # Randomly generate the dimension of the input tensor.
        dim = random.randint(1, 4)
        # Randomly generate the number of elements for each dimension.
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list of input sizes for the tensor.
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Create a random tensor with the specified input size.
        input_tensor = torch.abs(torch.randn(input_size)) + 1e-5
        # Perform the in-place log2 operation.
        input_tensor.log2_()
        # Return the modified tensor.
        return input_tensor
