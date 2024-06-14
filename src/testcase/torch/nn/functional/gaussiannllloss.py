import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.gaussiannllloss)
class TorchNnFunctionalGaussiannlllossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gaussian_nll_loss_correctness(self):
    # Randomly generate input size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random input tensor
    input_tensor = torch.randn(input_size)
    # Generate random target tensor with the same size as input_tensor
    target_tensor = torch.randn(input_size)
    # Generate random variance tensor, ensure it's positive
    var_tensor = torch.rand(input_size) + 1e-6

    # Calculate Gaussian NLL loss
    result = torch.nn.functional.gaussian_nll_loss(input_tensor, target_tensor, var_tensor)

    return result
