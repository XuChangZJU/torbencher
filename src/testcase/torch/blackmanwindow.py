import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.blackmanwindow)
class TorchBlackmanwindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_blackman_window_correctness(self):
        # Randomly generate the window length (positive integer)
        window_length = random.randint(1, 10)
        
        # Randomly generate the periodic flag (boolean value)
        periodic = random.choice([True, False])
        
        result = torch.blackman_window(window_length, periodic)
        return result
    