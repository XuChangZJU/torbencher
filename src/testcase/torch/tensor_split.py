
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.tensor_split)
class TorchTensorSplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensor_split_correctness(self):
        tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        indices_or_sections = random.randint(1, 10)
        dim = random.randint(0, 1)
        result = torch.tensor_split(tensor, indices_or_sections, dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_tensor_split_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        indices_or_sections = random.randint(1000, 10000)
        dim = random.randint(0, 1)
        result = torch.tensor_split(tensor, indices_or_sections, dim)
        return result

