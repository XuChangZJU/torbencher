import torch
import random
import torch.nn.utils.prune as prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.RandomStructured)
class TorchNnUtilsPruneRandomstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_structured_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(2, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor
        tensor = torch.randn(input_size)

        # Random amount to prune (either as a fraction or an absolute number)
        if random.choice([True, False]):
            amount = random.uniform(0.1, 0.9)  # Fraction of parameters to prune
        else:
            amount = random.randint(1, num_of_elements_each_dim)  # Absolute number of parameters to prune

        # Random dimension along which to prune
        prune_dim = random.randint(0, dim - 1)

        # Apply RandomStructured pruning
        pruner = prune.RandomStructured(amount, prune_dim)
        pruner.apply(tensor)

        return tensor
