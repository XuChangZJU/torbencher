import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.clamp_)
class TorchTensorClampUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clamp__correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)  # Random tensor
        min_val = random.uniform(-10.0, 10.0)  # Random min value
        max_val = random.uniform(min_val, 10.0)  # Random max value, ensuring max_val >= min_val
        input_tensor.clamp_(min_val, max_val)
        return input_tensor
