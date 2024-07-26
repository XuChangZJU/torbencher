import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MarginRankingLoss)
class TorchNnMarginrankinglossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_margin_ranking_loss_correctness(self):
        # Randomly choose the batch size
        batch_size = random.randint(1, 5)
        
        # Generate random tensors for input1, input2, and target
        input1 = torch.randn(batch_size, requires_grad=True)
        input2 = torch.randn(batch_size, requires_grad=True)
        
        # Generate random target tensor with values -1 or 1
        target = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
        
        # Create MarginRankingLoss criterion with default margin
        criterion = torch.nn.MarginRankingLoss()
        
        # Compute the loss
        loss = criterion(input1, input2, target)
        
        return loss
    
    
    
    