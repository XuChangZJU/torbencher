import torch
import torch.nn as nn
import torch.distributed as dist
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.parallel.DistributedDataParallel)
class TorchNnParallelDistributeddataparallelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_distributed_data_parallel_correctness(self):
        # Initialize the process group
        dist.init_process_group(backend='nccl', world_size=1, init_method='tcp://127.0.0.1:29500')

        # Randomly generate model parameters
        input_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        model = nn.Linear(input_size[-1], random.randint(1, 5)).cuda()

        # Wrap the model with DistributedDataParallel
        ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0)

        # Generate random input tensor
        input_tensor = torch.randn(input_size).cuda()

        # Forward pass
        output = ddp_model(input_tensor)

        return output
