
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.DisableTorchFunctionSubclass)
class TorchDisableTorchFunctionSubclassTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_disabletorchfunctionsubclass_correctness(self):
        result = torch.DisableTorchFunctionSubclass()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_disabletorchfunctionsubclass_large_scale(self):
        result = torch.DisableTorchFunctionSubclass()
        return result

