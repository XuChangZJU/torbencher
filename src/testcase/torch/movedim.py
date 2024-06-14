import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.movedim)
class TorchMovedimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_movedim_correctness(self):
    dim = random.randint(1, 4)
    input_size = [random.randint(1, 5) for _ in range(dim)]
    input_tensor = torch.randn(input_size)
    source = random.sample(range(dim), random.randint(1, dim))  # Generate unique source indices
    destination = random.sample(range(dim), len(source))  # Generate unique destination indices
    result = torch.movedim(input_tensor, source, destination)
    return result
