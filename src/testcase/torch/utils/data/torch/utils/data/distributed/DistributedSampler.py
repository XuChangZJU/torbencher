import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.torch.utils.data.distributed.DistributedSampler)
class TorchUtilsDataTorchUtilsDataDistributedDistributedsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_distributed_sampler_correctness(self):
        # Random parameters for DistributedSampler
        num_replicas = random.randint(1, 4)  # Number of processes participating in distributed training
        rank = random.randint(0, num_replicas - 1)  # Rank of the current process
        shuffle = random.choice([True, False])  # Whether to shuffle the indices or not
        seed = random.randint(0, 100)  # Random seed for reproducibility
        dataset_size = random.randint(10, 100)  # Size of the dataset
    
        # Create a dummy dataset
        dataset = torch.randn(dataset_size, 1)
    
        # Create a DistributedSampler instance
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
    
        # Get the sampled indices
        sampled_indices = list(sampler)
    
        return sampled_indices
    