import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.not_equal_)
class TorchTensorNotUequalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_not_equal__correctness(self):
        # Generate random dimension and size for the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors of the same size
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)

        # Perform in-place not_equal_ operation
        input_tensor.not_equal_(other_tensor)

        # Return the modified tensor
        return input_tensor
