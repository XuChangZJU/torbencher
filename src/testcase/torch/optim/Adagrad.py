import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.Adagrad)
class TorchOptimAdagradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_adagrad_correctness(self):
        # Define parameters for the optimizer
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        learning_rate = random.uniform(0.01, 0.1)
        lr_decay = random.uniform(0, 1)  # lr_decay should be in [0, 1)
        weight_decay = random.uniform(0, 1)
        eps = random.uniform(1e-10, 1e-6)

        # Create random tensor
        tensor = torch.randn(input_size, requires_grad=True)

        # Define optimizer
        optimizer = torch.optim.Adagrad(params=[tensor], lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay,
                                        eps=eps)

        # Perform optimization steps
        num_steps = random.randint(1, 10)
        for _ in range(num_steps):
            optimizer.zero_grad()
            output = tensor.sum()
            output.backward()
            optimizer.step()

        return tensor
