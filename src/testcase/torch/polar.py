import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.polar)
class TorchPolarTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_polar_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random absolute values (must be float or double)
        abs_tensor = torch.rand(input_size) + 0.1  # Ensure abs is always positive
        # Generate random angles
        angle_tensor = torch.randn(input_size)
        result = torch.polar(abs_tensor, angle_tensor)
        return result
