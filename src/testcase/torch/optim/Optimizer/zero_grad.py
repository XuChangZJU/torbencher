import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.Optimizer.zero_grad)
class TorchOptimOptimizerZerogradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_zero_grad_correctness(self):
        # Randomly generate the size of the model
        input_size = random.randint(1, 10)
        output_size = random.randint(1, 10)

        # Define a simple linear model
        model = torch.nn.Linear(input_size, output_size)

        # Define a random input tensor
        input_tensor = torch.randn((input_size,))

        # Define a random target tensor
        target_tensor = torch.randn((output_size,))

        # Define a loss function
        loss_fn = torch.nn.MSELoss()

        # Define an optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=random.uniform(0.01, 0.1))

        # Forward pass
        output = model(input_tensor)

        # Compute loss
        loss = loss_fn(output, target_tensor)

        # Backward pass
        loss.backward()

        # Check gradients before zero_grad
        gradients_before_zero_grad = [param.grad.clone() for param in model.parameters()]

        # Zero the gradients
        optimizer.zero_grad()

        # Check gradients after zero_grad
        gradients_after_zero_grad = [param.grad for param in model.parameters()]

        return gradients_before_zero_grad, gradients_after_zero_grad
