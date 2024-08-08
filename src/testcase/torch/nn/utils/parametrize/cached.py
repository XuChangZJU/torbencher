import random;

import torch;
import torch.nn as nn;
import torch.nn.utils.parametrize as P;

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase;
from src.util import test_api_version;


class TorchNnUtilsParametrizeCachedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cached_correctness(self):
        class SimpleModel(nn.Module):
            def __init__(self, linear_layer):
                super(SimpleModel, self).__init__();
                self.linear_layer = linear_layer;

            def forward(self, x):
                return self.linear_layer(x);

        # Load random input and output sizes
        in_features = random.randint(1, 10);
        out_features = random.randint(1, 10);

        # Create a random input tensor with matching size
        input_tensor = torch.randn((random.randint(1, 10), in_features));

        # Initialize the linear layer
        linear_layer = nn.Linear(in_features, out_features);
        linear_layer.weight = nn.Parameter(torch.normal(0, 1, size=linear_layer.weight.shape));
        linear_layer.bias = nn.Parameter(torch.normal(0, 1, size=linear_layer.bias.shape));
        P.register_parametrization(linear_layer, 'weight', nn.Identity());

        model = SimpleModel(linear_layer);

        # Forward pass without caching
        output_without_cache = model(input_tensor);

        # Forward pass with caching
        with P.cached():
            output_with_cache = model(input_tensor);

        return output_without_cache, output_with_cache;
