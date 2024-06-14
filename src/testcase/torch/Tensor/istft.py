import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.istft)
class TorchTensorIstftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_istft_correctness(self):
    # Randomly generate parameters for the test
    n_fft = random.randint(2, 10) * 2  # n_fft must be even
    hop_length = random.randint(1, n_fft // 2)  # hop_length must be <= n_fft // 2
    win_length = random.randint(1, n_fft)  # win_length must be <= n_fft
    window = torch.randn(win_length)  # Random window tensor of size win_length

    # Randomly generate a complex tensor for the input
    num_frames = random.randint(1, 10)
    num_freq_bins = n_fft // 2 + 1
    complex_tensor = torch.randn(num_frames, num_freq_bins, 2)  # Last dimension for real and imaginary parts

    # Perform the inverse short-time Fourier transform
    result = torch.istft(complex_tensor, n_fft, hop_length, win_length, window)
    return result
