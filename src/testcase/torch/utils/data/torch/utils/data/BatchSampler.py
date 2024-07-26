import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.torch.utils.data.BatchSampler)
class TorchUtilsDataTorchUtilsDataBatchsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_batch_sampler_correctness(self):
        # Randomly generate parameters for BatchSampler
        batch_size = random.randint(1, 10)  # Random batch size
        num_samples = random.randint(10, 50)  # Random number of samples
        drop_last = random.choice([True, False])  # Randomly choose whether to drop the last batch
    
        # Create a list of indices representing the dataset
        indices = list(range(num_samples))
    
        # Create a BatchSampler instance
        sampler = torch.utils.data.sampler.SequentialSampler(indices)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
    
        # Iterate over the batches and return the last batch
        for batch in batch_sampler:
            result = batch
        return result
    