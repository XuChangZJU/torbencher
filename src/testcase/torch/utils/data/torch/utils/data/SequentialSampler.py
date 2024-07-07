import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.torch.utils.data.SequentialSampler)
class TorchUtilsDataTorchUtilsDataSequentialsamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sequential_sampler_correctness(self):
        # Generate random parameters for the input data size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create random data
        data_source = torch.randn(input_size)
    
        # Create a SequentialSampler
        sampler = torch.utils.data.SequentialSampler(data_source)
    
        # Iterate over the sampler and return the sampled indices
        sampled_indices = list(sampler)
        return sampled_indices
    
    
    
    