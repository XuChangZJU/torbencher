import torch
import torch.nn.functional as F
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.nll_loss)
class TorchNnFunctionalNlllossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nll_loss_correctness(self):
        # Randomly choose dimensions for the input tensor
        N = random.randint(1, 4)  # Batch size
        C = random.randint(2, 10)  # Number of classes
        K = random.randint(1, 3)  # Number of additional dimensions (1D, 2D, or 3D loss)

        # Generate random input tensor with log-probabilities
        input_size = [N, C] + [random.randint(1, 5) for _ in range(K)]
        input_tensor = torch.randn(input_size, requires_grad=True)
        log_probs = F.log_softmax(input_tensor, dim=1)

        # Generate random target tensor with appropriate size and values
        target_size = [N] + input_size[2:]
        target_tensor = torch.randint(0, C, target_size)

        # Compute the negative log likelihood loss
        result = F.nll_loss(log_probs, target_tensor)
        return result
