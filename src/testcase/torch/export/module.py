import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.module)
class TorchExportModuleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_module_correctness(self):
        # Randomly generate the input tensor dimensions
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor with the generated dimensions
        input_tensor = torch.randn(input_size)

        # Define a simple model for testing
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = torch.nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Export the model using torch.jit.trace
        traced_model = torch.jit.trace(model, input_tensor)

        return traced_model
