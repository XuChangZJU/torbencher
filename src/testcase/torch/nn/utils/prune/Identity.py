import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.Identity)
class TorchNnUtilsPruneIdentityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_identity_pruning_correctness(self):
        # Randomly generate dimensions for the Linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a Linear layer with random dimensions
        linear_layer = nn.Linear(in_features, out_features)

        # Apply identity pruning to the 'weight' parameter of the Linear layer
        pruned_layer = prune.identity(linear_layer, 'weight')

        # Check if the mask and original weight are correctly added
        weight_mask = pruned_layer.weight_mask
        weight_orig = pruned_layer.weight_orig

        # Return the mask and original weight for verification
        return weight_mask, weight_orig
