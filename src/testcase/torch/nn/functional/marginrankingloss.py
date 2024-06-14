import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.marginrankingloss)
class TorchNnFunctionalMarginrankinglossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_margin_ranking_loss_correctness(self):
        # Define the dimensions for the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensors
        input1 = torch.randn(input_size)
        input2 = torch.randn(input_size)
    
        # Generate random target tensor with values -1 or 1
        target = torch.randint(0, 2, input_size) * 2 - 1 
    
        # Calculate the margin ranking loss
        loss = torch.nn.functional.margin_ranking_loss(input1, input2, target)
    
        return loss
    