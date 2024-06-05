
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.init_process_group)
class TorchInitProcessGroupTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_init_process_group_correctness(self):
        backend = random.choice([torch.distributed.Backend.GLOO, torch.distributed.Backend.NCCL, torch.distributed.Backend.MPI, torch.distributed.Backend.UCC])
        init_method = 'env://'
        world_size = random.randint(2, 10)
        rank = random.randint(0, world_size - 1)
        result = torch.distributed.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_init_process_group_large_scale(self):
        backend = random.choice([torch.distributed.Backend.GLOO, torch.distributed.Backend.NCCL, torch.distributed.Backend.MPI, torch.distributed.Backend.UCC])
        init_method = 'env://'
        world_size = random.randint(100, 1000)
        rank = random.randint(0, world_size - 1)
        result = torch.distributed.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
        return result

