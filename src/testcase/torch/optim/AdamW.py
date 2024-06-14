import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.AdamW)
class TorchOptimAdamwTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adamw_correctness(self):
    # Define the parameters to be optimized
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    weights = torch.randn(input_size, requires_grad=True)

    # Define a simple loss function
    def loss_fn(weights):
        return (weights ** 2).sum()
