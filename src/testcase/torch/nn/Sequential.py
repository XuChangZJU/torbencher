
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Sequential)
class TorchSequentialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sequential_correctness(self):
        modules = [torch.nn.Linear(random.randint(1, 10), random.randint(1, 10)), torch.nn.ReLU()]
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        sequential = torch.nn.Sequential(*modules)
        result = sequential(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_sequential_large_scale(self):
        modules = [torch.nn.Linear(random.randint(100, 1000), random.randint(100, 1000)), torch.nn.ReLU()]
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        sequential = torch.nn.Sequential(*modules)
        result = sequential(input_tensor)
        return result

