import torch
import random
from torch.utils.checkpoint import checkpoint

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


# @test_api(torch.utils.checkpoint.checkpoint)
class TorchUtilsCheckpointCheckpointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_checkpoint_correctness(self):
        # Define a simple model to use with checkpointing
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        # Create a random input tensor
        # input_size = [random.randint(1, 5) for _ in range(2)]  # Random 2D tensor
        # input_tensor = torch.randn(input_size)

        # Create a random input tensor with the correct shape
        batch_size = random.randint(1, 5)
        input_tensor = torch.randn(batch_size, 10)  # Ensure the last dimension is 10

        # Instantiate the model
        model = SimpleModel()

        # Define a function to be checkpointed
        def checkpointed_function(x):
            return model(x)

        # Use checkpointing
        result = checkpoint(checkpointed_function, input_tensor, use_reentrant=False)
        return result
