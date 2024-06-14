import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.BCEWithLogitsLoss)
class TorchNnBcewithlogitslossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bce_with_logits_loss_correctness(self):
    # Randomly generate batch size and number of classes
    batch_size = random.randint(1, 10)
    num_classes = random.randint(1, 10)
    
    # Generate random input tensor (logits) and target tensor
    input_tensor = torch.randn(batch_size, num_classes, requires_grad=True)
    target_tensor = torch.empty(batch_size, num_classes).random_(2)  # Binary targets (0 or 1)
    
    # Create BCEWithLogitsLoss criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Compute the loss
    loss = criterion(input_tensor, target_tensor)
    
    return loss
