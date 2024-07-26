import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.clip)
class TorchClipTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_clip_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)
        min_val = random.uniform(-10.0, 0.0)  # Random min_val value between -10.0 and 0.0
        max_val = random.uniform(0.0, 10.0)  # Random max_val value between 0.0 and 10.0
        result = torch.clip(input_tensor, min_val, max_val)
        return result
