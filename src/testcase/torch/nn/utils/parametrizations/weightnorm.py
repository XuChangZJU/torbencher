import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.parametrizations.weightnorm)
class TorchNnUtilsParametrizationsWeightnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_weight_norm_correctness(self):
    # Define the dimensions for the Linear layer
    in_features = random.randint(1, 10)
    out_features = random.randint(1, 10)

    # Create a Linear layer
    linear_layer = torch.nn.Linear(in_features, out_features)

    # Apply weight normalization to the Linear layer
    weight_normed_linear_layer = torch.nn.utils.parametrizations.weight_norm(linear_layer, name='weight')

    # Generate random input data
    input_data = torch.randn(in_features)

    # Perform inference using both the original and weight-normed layers
    output_original = linear_layer(input_data)
    output_weight_normed = weight_normed_linear_layer(input_data)

    return output_original, output_weight_normed
