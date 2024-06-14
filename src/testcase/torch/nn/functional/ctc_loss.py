import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.ctc_loss)
class TorchNnFunctionalCtclossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss_correctness(self):
        # Randomly generate input size
        batch_size = random.randint(1, 4)
        input_length = random.randint(1, 10)
        alphabet_size = random.randint(2, 10)  # Alphabet size should be at least 2 (including blank)
        max_target_length = random.randint(1, input_length)
    
        # Generate random input tensors
        log_probs = torch.randn(input_length, batch_size, alphabet_size).log_softmax(2).detach().requires_grad_()
        targets = torch.randint(1, alphabet_size, (batch_size, max_target_length), dtype=torch.long)
        input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
        target_lengths = torch.randint(1, max_target_length + 1, (batch_size,), dtype=torch.long)
    
        # Calculate CTC loss
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss
    
    
    
    