import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.huberloss)
class TorchNnFunctionalHuberlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_huber_loss_correctness(self):
    """
    Test the correctness of torch.nn.functional.huber_loss with random parameters.
    """
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    input = torch.randn(input_size) # Random input tensor
    target = torch.randn(input_size) # Random target tensor
    reduction = random.choice(['none', 'mean', 'sum']) # Randomly choose a reduction method
    delta = random.uniform(0.1, 10.0) # Random delta value between 0.1 and 10.0

    loss = torch.nn.functional.huber_loss(input, target, reduction=reduction, delta=delta)
    return loss
