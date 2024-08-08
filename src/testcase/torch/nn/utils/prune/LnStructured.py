import random

import torch.nn as nn
import torch.nn.utils.prune as prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(prune.LnStructured)
class TorchNnUtilsPruneLnstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_LnStructured_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(2, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a simple model with Linear layers
        model = nn.Sequential(nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim),
                              nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim))

        # Random amount to prune (either float between 0.0 and 1.0 or an integer)
        if random.choice([True, False]):
            amount = random.uniform(0.0, 1.0)
        else:
            amount = random.randint(1, num_of_elements_each_dim)

        # Random n value for L-norm
        n = random.choice([1, 2, float('inf'), float('-inf'), 'fro', 'nuc'])

        # Random dimension along which to prune
        prune_dim = random.randint(0, dim - 1)

        # Apply LnStructured pruning to the first layer's weights
        prune.ln_structured(model[0], name='weight', amount=amount, n=n, dim=prune_dim)

        # Here you can check the pruned weights or do other operations with the model
        pruned_weights = model[0].weight

        return pruned_weights
