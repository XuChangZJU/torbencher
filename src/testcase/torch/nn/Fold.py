
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Fold)
class TorchFoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fold_correctness(self):
        input_size = (random.randint(1, 10), random.randint(1, 10))
        output_size = (random.randint(1, 10), random.randint(1, 10))
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10) * random.randint(1, 10) * random.randint(1, 10) * random.randint(1, 10))
        fold = torch.nn.Fold(output_size, input_size)
        result = fold(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_fold_large_scale(self):
        input_size = (random.randint(100, 1000), random.randint(100, 1000))
        output_size = (random.randint(100, 1000), random.randint(100, 1000))
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000) * random.randint(100, 1000) * random.randint(100, 1000) * random.randint(100, 1000))
        fold = torch.nn.Fold(output_size, input_size)
        result = fold(input_tensor)
        return result

