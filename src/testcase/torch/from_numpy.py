
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.from_numpy)
class TorchFromNumpyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_from_numpy_correctness(self):
        import numpy as np
        numpy_array = np.random.randn(random.randint(1, 10))
        result = torch.from_numpy(numpy_array)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_from_numpy_large_scale(self):
        import numpy as np
        numpy_array = np.random.randn(random.randint(1000, 10000))
        result = torch.from_numpy(numpy_array)
        return result

