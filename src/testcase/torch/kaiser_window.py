import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.kaiser_window)
class TorchKaiserUwindowTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kaiser_window_correctness(self):
        # window_length: length of the window.
        window_length = random.randint(2, 10)  # window_length > 1
        # periodic: If True, returns a periodic window suitable for use in spectral analysis.
