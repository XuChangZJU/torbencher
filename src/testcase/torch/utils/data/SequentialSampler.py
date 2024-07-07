import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.SequentialSampler)
class TorchUtilsDataSequentialsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sequential_sampler_correctness(self):
        # Generate random parameters for the dataset size
        dataset_size = random.randint(1, 10)
    
        # Create a list to represent the dataset
        data_source = list(range(dataset_size))
    
        # Create a SequentialSampler
        sampler = torch.utils.data.SequentialSampler(data_source)
    
        # Get the sampled indices
        sampled_indices = list(sampler)
    
        # Return the sampled indices
        return sampled_indices
    
    
    
    