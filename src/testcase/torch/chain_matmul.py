
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.chain_matmul)
class TorchChainMatmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chain_matmul_correctness(self):
        num_tensors = random.randint(2, 10)
        dim = random.randint(1, 10)
        tensors = [torch.randn(dim, dim) for _ in range(num_tensors)]
        result = torch.chain_matmul(*tensors)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_chain_matmul_large_scale(self):
        num_tensors = random.randint(2, 10)
        dim = random.randint(100, 1000)
        tensors = [torch.randn(dim, dim) for _ in range(num_tensors)]
        result = torch.chain_matmul(*tensors)
        return result

