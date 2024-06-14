import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hannwindow)
class TorchHannwindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hann_window_correctness(self):
        # Random window length for the Hann window
        window_length = random.randint(1, 10)
        
        # Randomly set periodic parameter value: either True or False
        periodic = random.choice([True, False])
        
        # Create a Hann window with random parameters
        result = torch.hann_window(window_length, periodic)
        
        return result
    