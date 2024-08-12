import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.set_fusion_strategy)
class TorchJitSetUfusionUstrategyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_set_fusion_strategy_correctness(self):
        # Randomly generate fusion strategy
        strategy_type = random.choice(["STATIC", "DYNAMIC"])
        strategy_depth = random.randint(1, 5)
        fusion_strategy = [(strategy_type, strategy_depth)]

        # Set the fusion strategy
        torch.jit.set_fusion_strategy(fusion_strategy)

        # Define a simple model
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        # Create an instance of the model
        model = MyModel()

        # Trace the model
        traced_model = torch.jit.trace(model, torch.randn(1, 10))

        # Run an inference
        result = traced_model(torch.randn(1, 10))

        return result
