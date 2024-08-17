import random
import torch

from torch.utils.checkpoint import checkpoint
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util.decorator import test_api
from src.util import test_api_version

@test_api(torch.utils.checkpoint.checkpoint)
class TorchUtilsCheckpointCheckpointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
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

                # Initialize the weights and biases using normal distribution
                # First create a tensor of the appropriate shape, then fill it with values from a normal distribution
                with torch.no_grad():
                    self.linear1.weight.data.normal_(mean=0.0, std=1.0)
                    self.linear1.bias.data.normal_(mean=0.0, std=1.0)

                    self.linear2.weight.data.normal_(mean=0.0, std=1.0)
                    self.linear2.bias.data.normal_(mean=0.0, std=1.0)

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
        result = checkpoint(checkpointed_function, input_tensor, use_reentrant=False)
        return result
