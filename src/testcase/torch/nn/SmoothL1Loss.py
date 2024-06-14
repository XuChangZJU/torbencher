import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.SmoothL1Loss)
class TorchNnSmoothl1lossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_smooth_l1_loss_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size based on dimensions
    
        input_tensor = torch.randn(input_size)  # Random input tensor
        target_tensor = torch.randn(input_size)  # Random target tensor
        beta = random.uniform(0.1, 10.0)  # Random beta value between 0.1 and 10.0
    
        criterion = torch.nn.SmoothL1Loss(beta=beta)  # Initialize SmoothL1Loss with random beta
        loss = criterion(input_tensor, target_tensor)  # Compute the loss
    
        return loss
    
    
    
    