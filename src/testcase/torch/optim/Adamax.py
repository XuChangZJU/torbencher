import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.Adamax)
class TorchOptimAdamaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adamax_correctness(self):
        # Define the parameters for the Adamax optimizer
        learning_rate = random.uniform(1e-5, 1e-2)  # Random learning rate between 1e-5 and 1e-2
        beta1 = random.uniform(0.8, 0.99)  # Random beta1 value between 0.8 and 0.99
        beta2 = random.uniform(0.9, 0.999)  # Random beta2 value between 0.9 and 0.999
        weight_decay = random.uniform(0.0, 0.1)  # Random weight decay between 0.0 and 0.1

        # Create a simple linear model
        model = torch.nn.Linear(10, 1)

        # Create an Adamax optimizer
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, betas=(beta1, beta2),
                                       weight_decay=weight_decay)

        # Generate random input and target data
        input_data = torch.randn(100, 10)
        target_data = torch.randn(100, 1)

        # Define the loss function
        loss_function = torch.nn.MSELoss()

        # Train the model for a few epochs
        for epoch in range(10):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output_data = model(input_data)

            # Calculate the loss
            loss = loss_function(output_data, target_data)

            # Backward pass
            loss.backward()

            # Update the model parameters
            optimizer.step()

        return model.weight
