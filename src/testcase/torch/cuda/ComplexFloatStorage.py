
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.ComplexFloatStorage)
class TorchCudaComplexFloatStorageTestCase(TorBencherTestCaseBase):
    def test_complex_float_storage_correctness(self):
        size = random.randint(1, 10)
        data = [complex(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)) for _ in range(size)]
        result = torch.cuda.ComplexFloatStorage.from_list(data)
        return result

    def test_complex_float_storage_large_scale(self):
        size = random.randint(1000, 10000)
        data = [complex(random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)) for _ in range(size)]
        result = torch.cuda.ComplexFloatStorage.from_list(data)
        return result

