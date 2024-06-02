
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.stft)
class TorchStftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stft_4d(self, input=None):
        if input is not None:
            result = torch.stft(input[0], input[1], input[2], input[3])
            return [result, input]
        a = torch.randn(10)
        result = torch.stft(a, n_fft=4, hop_length=2, win_length=4)
        return [result, [a, 4, 2, 4]]

