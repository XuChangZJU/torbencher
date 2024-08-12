import random

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.ln_structured)
class TorchNnUtilsPruneLnUstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ln_structured_correctness(self):
        # Create a Conv2d module with random dimensions
        module = nn.Conv2d(5, 3, 2)

        # Randomly select the parameter name to prune
        param_name = 'weight'

        # Randomly generate the amount to prune (as a fraction)
        amount = random.uniform(0.1, 0.5)

        # Randomly select the norm type
        norm_type = random.choice([1, 2, float('inf'), float('-inf'), 'fro', 'nuc'])

        # Randomly select the dimension along which to prune
        dim = 0

        # Apply ln_structured pruning
        prune.ln_structured(module, param_name, amount, norm_type, dim)

        input_data = torch.randn(1, 5, 32, 32)

        # Return the pruned module to observe the effect
        return module(input_data)
