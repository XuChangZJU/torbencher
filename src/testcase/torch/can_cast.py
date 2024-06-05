
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.can_cast)
class TorchCanCastTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_can_cast_correctness(self):
        from_dtype = random.choice([torch.float32, torch.int32, torch.bool])
        to_dtype = random.choice([torch.float32, torch.int32, torch.bool])
        result = torch.can_cast(from_dtype, to_dtype)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_can_cast_large_scale(self):
        from_dtype = random.choice([torch.float32, torch.int32, torch.bool])
        to_dtype = random.choice([torch.float32, torch.int32, torch.bool])
        result = torch.can_cast(from_dtype, to_dtype)
        return result

