import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.remainder_)
class TorchTensorRemainderUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_remainder__correctness(self):
        # Randomly generate the dimension of the input tensor
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list of input sizes for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        input_tensor = torch.randn(input_size)
        # Generate a random divisor tensor with the same size as the input tensor
        divisor_tensor = torch.randn(input_size)
        # Perform the in-place remainder operation
        input_tensor.remainder_(divisor_tensor)
        # Return the modified input tensor
        return input_tensor
