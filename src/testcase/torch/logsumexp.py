import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logsumexp)
class TorchLogsumexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logsumexp_correctness(self):
    dim = random.randint(0, 3)  # Random dimension to reduce
    keepdim = random.choice([True, False]) # Randomly choose whether to keep the reduced dimension
    input_tensor = torch.randn(random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)) # Generate random input tensor
    result = torch.logsumexp(input_tensor, dim, keepdim)
    return result
