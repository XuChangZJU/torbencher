import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.Linear)
class TorchNnLinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linear_correctness(self):
        # Randomly generate valid parameters for torch.nn.Linear
        in_features = random.randint(1, 10)  # Random in_features between 1 and 10
        out_features = random.randint(1, 10)  # Random out_features between 1 and 10
        # Instantiate a Linear layer
        linear_layer = torch.nn.Linear(in_features, out_features)

        # initialize the parameters manually
        linear_layer.weight = torch.nn.Parameter(torch.normal(0, 0.01, size=(out_features, in_features)))
        linear_layer.bias = torch.nn.Parameter(torch.normal(0, 0.01, size=(out_features,)))

        # Randomly generate input tensor
        batch_size = random.randint(1, 10)  # Random batch size between 1 and 10
        input_tensor = torch.randn(batch_size, in_features)

        # Perform linear transformation
        output_tensor = linear_layer(input_tensor)

        return output_tensor
