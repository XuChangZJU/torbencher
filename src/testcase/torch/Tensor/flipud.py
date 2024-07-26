import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.flipud)
class TorchTensorFlipudTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_flipud_correctness(self):
        """Test the correctness of torch.Tensor.flipud."""
        dim = random.randint(2, 4)  # Dimension of the tensor should be at least 2 for flipud to have an effect
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)  # Generate a random tensor
        result = input_tensor.flipud()
        return result
