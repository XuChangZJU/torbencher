import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.NAdam)
class TorchOptimNadamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nadam_correctness(self):
        # Define the parameters for the optimizer
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        weight = torch.randn(input_size, requires_grad=True)
        lr = random.uniform(1e-5, 1e-2)  # Random learning rate
        betas = (random.uniform(0.8, 0.999), random.uniform(0.9, 0.999))  # Random betas
        eps = random.uniform(1e-9, 1e-7)  # Random epsilon
        weight_decay = random.uniform(0.0, 0.1)  # Random weight decay
        momentum_decay = random.uniform(1e-4, 1e-2)  # Random momentum decay

        # Create the optimizer
        optimizer = torch.optim.NAdam(
            [weight], lr, betas, eps, weight_decay, momentum_decay
        )

        # Perform a single optimization step
        output = weight * 2
        loss = output.mean()
        loss.backward()
        optimizer.step()

        return output
