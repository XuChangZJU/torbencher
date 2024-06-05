
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_default_tensor_type)
class TorchSetDefaultTensorTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_default_tensor_type_correctness(self):
        tensor_type = torch.FloatTensor
        result = torch.set_default_tensor_type(tensor_type)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_default_tensor_type_large_scale(self):
        tensor_type = torch.DoubleTensor
        result = torch.set_default_tensor_type(tensor_type)
        return result

