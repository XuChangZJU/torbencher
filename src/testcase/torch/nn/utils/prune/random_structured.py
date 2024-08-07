import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.random_structured)
class TorchNnUtilsPruneRandomUstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_structured_correctness(self):
        # Randomly generate dimensions for the Linear layer
        in_features = random.randint(2, 10)
        out_features = random.randint(2, 10)

        # Create a Linear layer with random dimensions
        module = nn.Linear(in_features, out_features)

        # Randomly choose the parameter name to prune
        param_name = 'weight'

        # Randomly generate the amount to prune
        amount = random.uniform(0.1, 0.9)  # Fraction of parameters to prune

        # Randomly choose the dimension along which to prune
        dim = random.randint(0, 1)

        # Apply random structured pruning
        pruned_module = prune.random_structured(module, param_name, amount, dim)

        # Check the number of pruned channels
        if dim == 0:
            pruned_channels = int(sum(torch.sum(pruned_module.weight, dim=1) == 0))
        else:
            pruned_channels = int(sum(torch.sum(pruned_module.weight, dim=0) == 0))

        return pruned_channels
