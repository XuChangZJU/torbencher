import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.istft)
class TorchIstftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_istft_correctness(self):
        # Random tensor dimensions
        batch_dim = random.randint(1, 3)  # Random batch dimension (optional)
        n_fft = random.randint(2, 10) * 2  # Ensure n_fft is even as FFT size
        hop_length = random.randint(1, n_fft)  # Random hop length within the range of n_fft
        win_length = random.randint(1, n_fft)  # Random window length within the range of n_fft
    
        # Generating the window tensor
        window = torch.randn(win_length)  # Random window function of window length
    
        # Number of frequency bins and frames
        num_frequency_bins = (n_fft // 2) + 1
        num_frames = random.randint(10, 20)  # Random number of frames
    
        # Creating the input tensor with random complex values
        input_tensor = torch.randn([batch_dim, num_frequency_bins, num_frames], dtype=torch.complex64)
    
        result = torch.istft(input_tensor, n_fft, hop_length, win_length, window)
        return result
    
    
    
    