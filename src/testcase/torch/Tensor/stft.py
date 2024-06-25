import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.stft)
class TorchTensorStftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stft_correctness(self):
        # Randomly generate the size of the input tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Generate a random input tensor
        input_tensor = torch.randn(input_size)
        
        # Randomly generate parameters for stft
        n_fft = random.randint(2, 10)  # FFT window size
        hop_length = random.randint(1, n_fft)  # Hop length
        win_length = random.randint(1, n_fft)  # Window length
        
        # Generate a random window tensor if win_length is specified
        window = torch.randn(win_length) if win_length else None
        
        # Ensure the input tensor has at least 2 dimensions for STFT
        if input_tensor.dim() < 2:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Perform Short-time Fourier Transform (STFT)
        result = torch.stft(input_tensor, n_fft, hop_length, win_length, window)
        
        return result
    
    
    
    