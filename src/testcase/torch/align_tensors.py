
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.align_tensors)
class TorchAlignTensorsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_align_tensors_correctness(self):
        dim1 = random.randint(1, 10)
        dim2 = random.randint(1, 10)
        tensor1 = torch.randn(dim1)
        tensor2 = torch.randn(dim2)
        result = torch.align_tensors(tensor1, tensor2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_align_tensors_large_scale(self):
        dim1 = random.randint(1000, 10000)
        dim2 = random.randint(1000, 10000)
        tensor1 = torch.randn(dim1)
        tensor2 = torch.randn(dim2)
        result = torch.align_tensors(tensor1, tensor2)
        return result

