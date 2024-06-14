import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.SoftMarginLoss)
class TorchNnSoftmarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_soft_margin_loss_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
    input_size = [num_of_elements_each_dim for _ in range(dim)]

    input_tensor = torch.randn(input_size)  # Random input tensor
    target_tensor = torch.randint(0, 2, input_size).float() * 2 - 1  # Random target tensor with values -1 or 1

    criterion = torch.nn.SoftMarginLoss()  # Using default reduction 'mean'
    loss = criterion(input_tensor, target_tensor)
    return loss
