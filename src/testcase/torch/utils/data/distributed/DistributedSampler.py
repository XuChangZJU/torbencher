import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.distributed.DistributedSampler)
class TorchUtilsDataDistributedDistributedsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_DistributedSampler_correctness(self):
        # Random parameters for DistributedSampler
        num_replicas = random.randint(1, 4)  # Number of replicas
        rank = random.randint(0, num_replicas - 1)  # Rank of the current process
        shuffle = random.choice([True, False])  # Whether to shuffle the indices
        seed = random.randint(0, 100)  # Random seed for shuffling

        # Create a dataset
        dataset_size = random.randint(10, 50)
        dataset = torch.randn(dataset_size, 3)

        # Create a DistributedSampler
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank,
                                                                  shuffle=shuffle, seed=seed)

        # Get the sampled indices
        indices = list(sampler)

        return indices
