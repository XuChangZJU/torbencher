import random
import torch
from torch.utils.checkpoint import set_checkpoint_debug_enabled
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(set_checkpoint_debug_enabled)
class TorchUtilsCheckpointSetUcheckpointUdebugUenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_set_checkpoint_debug_enabled_correctness(self):
        # Randomly enable or disable checkpoint debug mode
        enable_debug = random.choice([True, False])

        # Define a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(SimpleModel, self).__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

                # Initialize the weights and biases using normal distribution
                with torch.no_grad():
                    self.linear.weight.data = torch.normal(0, 0.01, (out_features, in_features))
                    self.linear.bias.data = torch.normal(0, 0.01, (out_features,))

            def forward(self, x):
                return self.linear(x)

        # Initialize the model
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        model = SimpleModel(in_features, out_features)

        # Set checkpoint debug mode
        previous_state = set_checkpoint_debug_enabled(enable_debug)

        # Generate some random input data
        input_data = torch.randn(1, in_features)  # Example: Batch size of 1 and 10 features

        # Perform a forward pass with the model
        output = model(input_data)

        return previous_state, output
