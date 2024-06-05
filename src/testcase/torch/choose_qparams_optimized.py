
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.choose_qparams_optimized)
class TorchChooseQparamsOptimizedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_choose_qparams_optimized_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randn(dim)
        dtype = random.choice([torch.qint8, torch.quint8, torch.qint32])
        reduce_range = random.choice([True, False])
        result = torch.choose_qparams_optimized(input, dtype, reduce_range=reduce_range)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_choose_qparams_optimized_large_scale(self):
        dim = random.randint(1000, 10000)
        input = torch.randn(dim)
        dtype = random.choice([torch.qint8, torch.quint8, torch.qint32])
        reduce_range = random.choice([True, False])
        result = torch.choose_qparams_optimized(input, dtype, reduce_range=reduce_range)
        return result

