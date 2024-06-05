
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.gather_object)
class TorchGatherObjectTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gather_object_correctness(self):
        obj = random.randint(1, 100)
        result = torch.distributed.gather_object(obj)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_gather_object_large_scale(self):
        obj = random.randint(1, 100)
        result = torch.distributed.gather_object(obj)
        return result

