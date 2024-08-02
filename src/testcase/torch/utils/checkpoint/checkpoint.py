import torch
import random
from torch.utils.checkpoint import checkpoint

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version


class TorchUtilsCheckpointCheckpointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_checkpoint_correctness(self):
        # Generate random sizes for the layers
        input_features = random.randint(1, 10)
        hidden_features = random.randint(1, 10)
        output_features = random.randint(1, 10)
        
        # Define a simple model to use with checkpointing
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear1 = torch.nn.Linear(input_features, hidden_features)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(hidden_features, output_features)

                # Manually initialize the weights and biases with torch.normal
                self.linear1.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=self.linear1.weight.shape))
                self.linear1.bias = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=self.linear1.bias.shape))

                self.linear2.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=self.linear2.weight.shape))
                self.linear2.bias = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=self.linear2.bias.shape))

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        # Create a random input tensor
        batch_size = random.randint(1, 5)  # Random batch size
        input_tensor = torch.randn(batch_size, input_features)

        # Instantiate the model
        model = SimpleModel()

        # Define a function to be checkpointed
        def checkpointed_function(x):
            return model(x)

        # Use checkpointing
        result = checkpoint(checkpointed_function, input_tensor)
        return result
