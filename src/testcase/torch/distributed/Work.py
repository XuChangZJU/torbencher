
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.Work)
class TorchWorkTestCase3(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_work_exception_correctness(self):
        result = torch.distributed.Work.exception()
        return result

    @test_api_version.larger_than("1