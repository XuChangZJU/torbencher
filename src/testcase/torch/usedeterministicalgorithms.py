import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.usedeterministicalgorithms)
class TorchUsedeterministicalgorithmsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_use_deterministic_algorithms_correctness(self):
    # mode: whether to use deterministic algorithms
    mode = random.choice([True, False]) 
    # warn_only: whether to throw a warning instead of an error if a deterministic implementation is not available
    warn_only = random.choice([True, False]) 
    result = torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    return result
