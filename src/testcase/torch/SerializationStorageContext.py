
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.SerializationStorageContext)
class TorchSerializationStorageContextTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_serializationstoragecontext_correctness(self):
        result = torch.SerializationStorageContext()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_serializationstoragecontext_large_scale(self):
        result = torch.SerializationStorageContext()
        return result

