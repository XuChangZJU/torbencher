import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PoissonNLLLoss)
class TorchNnPoissonnlllossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_poisson_nll_loss_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(1, 4)
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for _ in range(dim)]

    # Random input tensor
    input_tensor = torch.randn(input_size, requires_grad=True)
    # Random target tensor with positive values to ensure valid Poisson distribution
    target_tensor = torch.abs(torch.randn(input_size))

    # Create PoissonNLLLoss instance with default parameters
    loss_fn = torch.nn.PoissonNLLLoss()

    # Compute the loss
    loss = loss_fn(input_tensor, target_tensor)
    return loss
