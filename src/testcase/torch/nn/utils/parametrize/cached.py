import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.parametrize.cached)
class TorchNnUtilsParametrizeCachedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cached_correctness(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(random.randint(1, 10), random.randint(1, 10))
                P.register_parametrization(self.linear, 'weight', nn.Identity())

            def forward(self, x):
                return self.linear(x)

        # Create a random input tensor
        input_size = [random.randint(1, 10) for _ in range(random.randint(1, 4))]
        input_tensor = torch.randn(input_size)

        model = SimpleModel()

        # Forward pass without caching
        output_without_cache = model(input_tensor)

        # Forward pass with caching
        with P.cached():
            output_with_cache = model(input_tensor)

        return output_without_cache, output_with_cache
