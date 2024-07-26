import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.smooth_l1_loss)
class TorchNnFunctionalSmoothl1lossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_smooth_l1_loss_correctness(self):
        # Define the dimensions for the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensors
        input = torch.randn(input_size)
        target = torch.randn(input_size)
    
        # Generate random beta value
        beta = random.uniform(0.1, 10.0)
    
        # Calculate the SmoothL1Loss
        result = torch.nn.functional.smooth_l1_loss(input, target, beta=beta)
        return result
    
    
    
    