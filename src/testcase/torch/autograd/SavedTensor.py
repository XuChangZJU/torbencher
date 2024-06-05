
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.SavedTensor)
class TorchAutogradSavedTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_savedtensor_correctness(self):
        input = torch.randn(random.randint(1, 10), requires_grad=True)
        saved_tensor = torch.autograd.SavedTensor(input)
        return saved_tensor

    @test_api_version.larger_than("1.1.3")
    def test_savedtensor_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), requires_grad=True)
        saved_tensor = torch.autograd.SavedTensor(input)
        return saved_tensor


