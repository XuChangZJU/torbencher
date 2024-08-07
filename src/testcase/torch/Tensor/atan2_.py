import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.atan2_)
class TorchTensorAtan2UTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atan2__correctness(self):
        # Generate random dimension and size for input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensors
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)

        # Apply atan2_ operation in-place
        input_tensor.atan2_(other_tensor)

        return input_tensor
