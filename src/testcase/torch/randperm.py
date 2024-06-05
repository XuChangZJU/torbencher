
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randperm)
class TorchRandpermTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randperm_correctness(self):
        n = random.randint(1, 10)
        result = torch.randperm(n)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_randperm_large_scale(self):
        n = random.randint(1000, 10000)
        result = torch.randperm(n)
        return result

@test_api(torch.quasirandom.SobolEngine)
class TorchQuasirandomSobolEngineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sobol_engine_correctness(self):
        result = torch.quasirandom.SobolEngine(dimension=random.randint(1, 10))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_sobol_engine_large_scale(self):
        result = torch.quasirandom.SobolEngine(dimension=random.randint(1000, 10000))
        return result

