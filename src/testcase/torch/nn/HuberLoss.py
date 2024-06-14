import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.HuberLoss)
class TorchNnHuberlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_huber_loss_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(1, 4)
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for _ in range(dim)]

    # Generate random input and target tensors
    input_tensor = torch.randn(input_size)
    target_tensor = torch.randn(input_size)
    
    # Random delta value between 0.1 and 10.0
    delta = random.uniform(0.1, 10.0)
    
    # Create HuberLoss criterion
    criterion = torch.nn.HuberLoss(delta=delta)
    
    # Compute the loss
    loss = criterion(input_tensor, target_tensor)
    
    return loss
