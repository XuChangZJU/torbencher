
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CTCLoss)
class TorchNNCTCLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss(self):
        
        log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        ctc_loss = torch.nn.CTCLoss()
        result = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return result

