import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.istft)
class TorchIstftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_istft_4d(self, input=None):
        if input is not None:
            result = torch.istft(input[0], input[1], input[2], input[3], input[4])
            return [result, input]
        a = torch.randn(3, 6, 2)  # Example shape for STFT output
        result = torch.istft(a, n_fft=4, hop_length=2, win_length=4)
        return [result, [a, 4, 2, 4, None]]

