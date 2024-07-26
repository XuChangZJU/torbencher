import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.BCELoss)
class TorchNnBcelossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bceloss_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Random input tensor with values between 0 and 1
        input_tensor = torch.sigmoid(torch.randn(input_size))
        # Random target tensor with values between 0 and 1
        target_tensor = torch.rand(input_size)
    
        # Create BCELoss instance
        bce_loss = torch.nn.BCELoss()
    
        # Calculate the loss
        result = bce_loss(input_tensor, target_tensor)
        return result
    
    
    
    