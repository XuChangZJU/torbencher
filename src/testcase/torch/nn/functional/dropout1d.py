import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.dropout1d)
class TorchNnFunctionalDropout1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout1d_correctness(self):
        batch_size = random.randint(1, 4)  # Random batch size
        num_channels = random.randint(1, 4)  # Random number of channels
        length = random.randint(1, 10)  # Random length of each channel
    
        input_tensor = torch.randn(batch_size, num_channels, length)
        p = random.uniform(0.1, 0.9)  # Random dropout probability between 0.1 and 0.9
        result = torch.nn.functional.dropout1d(input_tensor, p)
        return result
    
    
    
    