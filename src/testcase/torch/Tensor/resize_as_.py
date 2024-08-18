import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.resize_as_)
class TorchTensorResizeUasUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_resize_as_correctness(self):
        dim = 4  # Random dimension for the tensors
        num_of_elements_each_dim = 5  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        input_tensor = torch.randn(input_size)
        target_size = [random.randint(1, 10) for _ in range(dim)]  # Generate random target size
        target_tensor = torch.randn(target_size)

        result = input_tensor.resize_as_(target_tensor)
        return result.shape
