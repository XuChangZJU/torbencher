
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.DeserializationStorageContext)
class TorchDeserializationStorageContextTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_deserializationstoragecontext_correctness(self):
        result = torch.DeserializationStorageContext()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_deserializationstoragecontext_large_scale(self):
        result = torch.DeserializationStorageContext()
        return result

