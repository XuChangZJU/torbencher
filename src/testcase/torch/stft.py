import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.stft)
class TorchStftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stft_correctness(self):
        # Random batch size and sequence length
        batch_size = random.randint(1, 4)
        seq_length = random.randint(20, 50)
        
        # Random input tensor size
        input_tensor = torch.randn(batch_size, seq_length)
        
        # Random n_fft value
        n_fft = random.randint(10, 20)
    
        # Perform STFT without additional keyword arguments
        stft_result = torch.stft(input_tensor, n_fft, return_complex=True)
        
        return stft_result
    