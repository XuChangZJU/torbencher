import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.Optimizer.step)
class TorchOptimOptimizerStepTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_optimizer_step_correctness(self):
        # Define the dimensions for the parameters
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create random parameters
        weight = torch.randn(input_size, requires_grad=True)
        bias = torch.randn(input_size, requires_grad=True)

        # Define a simple loss function
        def loss_fn(weight, bias):
            # Generate random input and target
            input_tensor = torch.randn(input_size)
            target = torch.randn(input_size)
            # Calculate output
            output = weight * input_tensor + bias
            # Use mean squared error loss
            loss = torch.mean((output - target) ** 2)
            return loss

        # Initialize an optimizer
        optimizer = torch.optim.SGD([weight, bias], lr=0.1)

        # Perform a single optimization step
        loss = loss_fn(weight, bias)
        loss.backward()
        optimizer.step()
        return weight
