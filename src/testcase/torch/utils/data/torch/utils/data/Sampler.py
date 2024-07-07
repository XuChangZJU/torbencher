import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.torch.utils.data.Sampler)
class TorchUtilsDataTorchUtilsDataSamplerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sampler_correctness(self):
        # Randomly generate the length of data
        data_length = random.randint(1, 10)
        # Create a list of indices
        indices = list(range(data_length))
        # Create a Sampler object
        sampler = torch.utils.data.Sampler(indices)
        # Convert the sampler object to a list
        result = list(sampler)
        return result
    
    
    
    