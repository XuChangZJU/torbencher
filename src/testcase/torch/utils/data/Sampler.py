import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.Sampler)
class TorchUtilsDataSamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sampler_correctness(self):
        # Generate random data
        data_size = random.randint(10, 20)
        data = [str(i) for i in range(data_size)]
    
        # Create a sampler
        class AccedingSequenceLengthSampler(torch.utils.data.Sampler):
            def __init__(self, data):
                super().__init__(data)
                self.data = data
    
            def __len__(self):
                return len(self.data)
    
            def __iter__(self):
                sizes = torch.tensor([len(x) for x in self.data])
                yield from torch.argsort(sizes).tolist()
    
        sampler = AccedingSequenceLengthSampler(data)
    
        # Get the sampled indices
        sampled_indices = list(sampler)
    
        # Check if the sampled indices are in ascending order of data length
        assert all(len(data[sampled_indices[i]]) <= len(data[sampled_indices[i+1]]) for i in range(len(sampled_indices)-1))
    
        return sampled_indices
    
    
    
    