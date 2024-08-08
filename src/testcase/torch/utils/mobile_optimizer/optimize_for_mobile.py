import random

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version


class TorchUtilsMobileUoptimizerOptimizeUforUmobileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_optimize_for_mobile_correctness(self):
        # Generate random sizes for the layers
        input_features = random.randint(1, 5)
        hidden_features = random.randint(1, 10)
        output_features = random.randint(1, 10)

        # Create the layers with the randomly generated sizes
        L1 = torch.nn.Linear(input_features, hidden_features)
        L2 = torch.nn.Linear(hidden_features, output_features)

        # Manually initialize the weights and biases with torch.normal
        L1.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=L1.weight.shape))
        L1.bias = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=L1.bias.shape))

        L2.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=L2.weight.shape))
        L2.bias = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=L2.bias.shape))

        model = torch.nn.Sequential(L1, torch.nn.ReLU(), L2)

        # Generate a random input tensor
        batch_size = random.randint(1, 4)
        input_tensor = torch.randn(batch_size, input_features)

        # Optimize the model for mobile
        scripted_model = torch.jit.script(model)
        optimized_model = optimize_for_mobile(scripted_model)

        # Run the model and the optimized model with the same input
        original_output = model(input_tensor)
        optimized_output = optimized_model(input_tensor)

        return original_output, optimized_output
