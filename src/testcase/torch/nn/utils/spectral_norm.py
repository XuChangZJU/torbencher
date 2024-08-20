import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.spectral_norm)
class TorchNnUtilsSpectralUnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_spectral_norm_correctness(self):
        # Define the input size for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a Linear layer
        linear_layer = torch.nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear_layer.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
            linear_layer.bias = torch.nn.Parameter(torch.randn(out_features))

        # Apply spectral normalization to the Linear layer
        spectral_norm_linear = torch.nn.utils.spectral_norm(linear_layer)

        # Generate random input data
        input_data = torch.randn(random.randint(1, 10), in_features)

        # Perform inference using both the original and spectral normalized layers
        output_original = linear_layer(input_data)
        output_spectral_norm = spectral_norm_linear(input_data)

        # Return the output from the spectral normalized layer
        return torch.allclose(output_original, output_spectral_norm)
