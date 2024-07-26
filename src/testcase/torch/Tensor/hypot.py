import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.hypot)
class TorchTensorHypotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hypot_correctness(self):
        # Generate random dimension and size for input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensors with the same size
        input_tensor = torch.randn(input_size)
        other_tensor = torch.randn(input_size)

        # Calculate hypot
        result = input_tensor.hypot(other_tensor)
        return result
