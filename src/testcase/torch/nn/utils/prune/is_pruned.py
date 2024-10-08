import random

import torch
from torch.nn.utils import prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.is_pruned)
class TorchNnUtilsPruneIsUprunedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_is_pruned_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor
        tensor = torch.randn(input_size)

        # Create a module
        module = torch.nn.Linear(tensor.shape[0], 5)

        # Check if the module is pruned (should be False)
        result1 = prune.is_pruned(module)

        # Apply pruning
        prune.random_unstructured(module, name="weight", amount=0.2)

        # Check if the module is pruned (should be True)
        result2 = prune.is_pruned(module)

        return result1, result2
