import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.L1Loss)
class TorchNnL1lossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_L1Loss_correctness(self):
    # Define the dimensions of the input tensors
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random input tensors
    input_tensor = torch.randn(input_size, requires_grad=True)
    target_tensor = torch.randn(input_size)

    # Create an instance of the L1Loss module
    l1_loss = torch.nn.L1Loss()

    # Calculate the L1 loss
    output_loss = l1_loss(input_tensor, target_tensor)

    # Return the calculated loss
    return output_loss
