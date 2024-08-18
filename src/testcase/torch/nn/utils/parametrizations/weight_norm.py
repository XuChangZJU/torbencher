import random
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(parametrizations.weight_norm)
class TorchNnUtilsParametrizationsWeightUnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_weight_norm_correctness(self):
        # Define the dimensions for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a Linear layer
        linear_layer = nn.Linear(in_features, out_features)

        # Initialize the weights and biases of the linear layer
        with torch.no_grad():
            linear_layer.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)
            if linear_layer.bias is not None:
                linear_layer.bias = torch.nn.Parameter(torch.randn(out_features) * 0.01)

        # Apply weight normalization to the Linear layer
        weight_normed_linear_layer = parametrizations.weight_norm(linear_layer, name='weight')

        # Generate random input data
        input_data = torch.randn(1, in_features)  # Add batch dimension

        # Perform inference using both the original and weight-normed layers
        output_original = linear_layer(input_data)
        output_weight_normed = weight_normed_linear_layer(input_data)

        return output_original, output_weight_normed
