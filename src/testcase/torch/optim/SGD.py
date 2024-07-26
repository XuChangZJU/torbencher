import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.SGD)
class TorchOptimSgdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sgd_correctness(self):
        # Randomly generate model parameters
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a simple model with random parameters
        model = torch.nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim)

        # Randomly generate learning rate
        lr = random.uniform(0.001, 0.1)

        # Randomly generate momentum
        momentum = random.uniform(0.0, 0.9)

        # Create SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum)

        # Generate random input and target tensors
        input_tensor = torch.randn(input_size)
        target_tensor = torch.randn(input_size)

        # Define a simple loss function
        loss_fn = torch.nn.MSELoss()

        # Forward pass
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Perform optimization step
        optimizer.step()

        return model.parameters()
