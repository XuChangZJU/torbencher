import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.random_unstructured)
class TorchNnUtilsPruneRandomunstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_unstructured_correctness(self):
        # Randomly generate dimensions for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a Linear layer with random dimensions
        module = nn.Linear(in_features, out_features)

        # Randomly choose the amount to prune
        amount = random.uniform(0.1, 0.9)  # Fraction of parameters to prune

        # Apply random unstructured pruning
        pruned_module = prune.random_unstructured(module, 'weight', amount)

        # Check the number of pruned elements
        pruned_elements = torch.sum(pruned_module.weight_mask == 0).item()

        return pruned_elements
