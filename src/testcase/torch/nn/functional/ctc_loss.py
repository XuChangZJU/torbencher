import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.ctc_loss)
class TorchNNFunctionalCTCLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ctc_loss_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.ctc_loss(input[0], input[1], input[2], input[3], input[4], input[5], input[6])
            return [result, input]
        log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        blank = 0
        reduction = 'mean'
        zero_infinity = False
        result = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)
        return [result, [log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity]]
