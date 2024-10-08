import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.acos)
class TorchAcosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_acos_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)  # Random tensor
        # normalizing the input tensor to be in the range of [-1, 1]
        input_tensor = torch.clamp(input_tensor, -1, 1)
        
        result = torch.acos(input_tensor)
        return result
