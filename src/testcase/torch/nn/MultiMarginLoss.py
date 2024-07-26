import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MultiMarginLoss)
class TorchNnMultimarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multi_margin_loss_correctness(self):
        # Randomly generate batch size and number of classes
        batch_size = random.randint(1, 5)
        num_classes = random.randint(2, 10)  # At least 2 classes
    
        # Generate random input tensor of shape (batch_size, num_classes)
        input_tensor = torch.randn(batch_size, num_classes)
    
        # Generate random target tensor of shape (batch_size) with class indices
        target_tensor = torch.randint(0, num_classes, (batch_size,))
    
        # Create MultiMarginLoss criterion with default parameters
        criterion = torch.nn.MultiMarginLoss()
    
        # Compute the loss
        loss = criterion(input_tensor, target_tensor)
        return loss
    
    
    
    