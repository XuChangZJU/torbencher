import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.soft_margin_loss)
class TorchNnFunctionalSoftUmarginUlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_soft_margin_loss_correctness(self):
        # Define the dimensions for the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        input = torch.randn(input_size)

        # Generate random target tensor with values -1 or 1
        target = torch.randint(0, 2, input_size) * 2 - 1

        # Calculate the soft margin loss
        loss = torch.nn.functional.soft_margin_loss(input, target)

        return loss
