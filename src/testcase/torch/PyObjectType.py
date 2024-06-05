
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.PyObjectType)
class TorchPyObjectTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pyobjecttype_correctness(self):
        result = torch.PyObjectType()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_pyobjecttype_large_scale(self):
        result = torch.PyObjectType()
        return result

