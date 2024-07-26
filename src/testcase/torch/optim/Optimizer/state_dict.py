import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.Optimizer.state_dict)
class TorchOptimOptimizerStatedictTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_optimizer_state_dict_correctness(self):
        # Randomly generate the number of parameters
        num_params = random.randint(1, 5)

        # Create a simple model with the generated number of parameters
        # Generate random sizes for each layer
        layer_sizes = [random.randint(1, 10) for _ in range(num_params + 1)]

        # Create a simple model with the generated sizes
        layers = [torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(num_params)]
        model = torch.nn.Sequential(*layers)

        # Randomly choose an optimizer
        optimizer_class = random.choice([torch.optim.SGD, torch.optim.Adam])

        # Randomly generate learning rate and weight decay
        lr = random.uniform(0.001, 0.1)
        weight_decay = random.uniform(0.0, 0.1)

        # Initialize the optimizer with the model parameters
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Perform a dummy forward and backward pass to initialize optimizer state
        # input_tensor = torch.randn((random.randint(1, 10), random.randint(1, 10)))
        input_tensor = torch.randn((random.randint(1, 10), layer_sizes[0]))  # Ensure input size matches first layer
        output_tensor = model(input_tensor)
        loss = output_tensor.sum()
        loss.backward()
        optimizer.step()

        # Get the state dictionary of the optimizer
        state_dict = optimizer.state_dict()

        return state_dict
