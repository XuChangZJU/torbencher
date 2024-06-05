
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.is_backend_available)
class TorchIsBackendAvailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_backend_available_correctness(self):
        backend = random.choice([torch.distributed.Backend.GLOO, torch.distributed.Backend.NCCL, torch.distributed.Backend.MPI, torch.distributed.Backend.UCC])
        result = torch.distributed.is_backend_available(backend)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_backend_available_large_scale(self):
        backend = random.choice([torch.distributed.Backend.GLOO, torch.distributed.Backend.NCCL, torch.distributed.Backend.MPI, torch.distributed.Backend.UCC])
        result = torch.distributed.is_backend_available(backend)
        return result

