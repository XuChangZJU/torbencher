import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.expm1_)
class TorchTensorExpm1UTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_expm1__correctness(self):
        # Randomly generate the dimension of the input tensor.
        dim = random.randint(1, 4)
        # Randomly generate the number of elements in each dimension.
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor.
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor.
        input_tensor = torch.randn(input_size)
        # Perform the operation.
        input_tensor.expm1_()
        # Return the tensor after the operation.
        return input_tensor
