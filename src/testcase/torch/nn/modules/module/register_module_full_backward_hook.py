import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.modules.module.register_module_full_backward_hook)
class TorchNnModulesModuleRegisterUmoduleUfullUbackwardUhookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_register_module_full_backward_hook_correctness(self):
        # Define a simple neural network module
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 2)
                with torch.no_grad():
                    self.fc1.weight = torch.nn.Parameter(torch.randn(5, 10))
                    self.fc1.bias = torch.nn.Parameter(torch.randn(5))
                with torch.no_grad():
                    self.fc2.weight = torch.nn.Parameter(torch.randn(2, 5))
                    self.fc2.bias = torch.nn.Parameter(torch.randn(2))

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Instantiate the network
        net = SimpleNet()

        # Define a backward hook function
        def backward_hook(module, grad_input, grad_output):
            # Optionally modify the gradients
            new_grad_input = tuple(g * random.uniform(0.5, 1.5) if g is not None else None for g in grad_input)
            return new_grad_input

        # Register the backward hook
        handle = torch.nn.modules.module.register_module_full_backward_hook(backward_hook)

        # Generate random input tensor
        input_tensor = torch.randn(1, 10, requires_grad=True)

        # Forward pass
        output = net(input_tensor)

        # Generate random target tensor
        target = torch.randn(1, 2)

        # Define a loss function
        loss_fn = torch.nn.MSELoss()

        # Compute loss
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Check gradients
        gradients = input_tensor.grad

        # Remove the hook
        handle.remove()

        return gradients
