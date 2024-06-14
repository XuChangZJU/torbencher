import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GaussianNLLLoss)
class TorchNnGaussiannlllossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gaussian_nll_loss_correctness(self):
        # Randomly generate dimensions and size for input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random input, target, and variance tensors
        input_tensor = torch.randn(input_size, requires_grad=True)
        target_tensor = torch.randn(input_size)
        var_tensor = torch.randn(input_size).abs()  # Ensure variance is positive
    
        # Initialize GaussianNLLLoss with default parameters
        loss_fn = torch.nn.GaussianNLLLoss()
    
        # Compute the loss
        loss = loss_fn(input_tensor, target_tensor, var_tensor)
        return loss
    
    
    
    