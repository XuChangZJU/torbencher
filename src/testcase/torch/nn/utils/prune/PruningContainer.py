import torch
import random
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(prune.PruningContainer)
class TorchNnUtilsPrunePruningcontainerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pruning_container_correctness(self):
        # Create a simple model with a Linear layer
        model = nn.Linear(10, 10)

        # Create two pruning methods
        pruning_method_1 = prune.L1Unstructured(amount=random.uniform(0.1, 0.5))
        pruning_method_2 = prune.RandomUnstructured(amount=random.uniform(0.1, 0.5))

        # Apply the pruning methods to the model's weight parameter
        prune.l1_unstructured(model, name='weight', amount=pruning_method_1.amount)
        prune.random_unstructured(model, name='weight', amount=pruning_method_2.amount)

        # Check the pruned weights
        pruned_weights = model.weight

        return pruned_weights
