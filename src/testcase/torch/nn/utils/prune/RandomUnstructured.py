import torch
import random
import torch.nn.utils.prune as prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.RandomUnstructured)
class TorchNnUtilsPruneRandomunstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_unstructured_correctness(self):
        # Randomly generate dimensions for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor
        tensor = torch.randn(input_size)

        # Create a module and assign the tensor to a parameter
        module = torch.nn.Linear(tensor.size(-1), tensor.size(-1))
        module.weight.data = tensor

        # Randomly generate the amount to prune
        if random.choice([True, False]):
            amount = random.uniform(0.1, 1.0)  # Fraction of parameters to prune
        else:
            amount = random.randint(1, tensor.numel())  # Absolute number of parameters to prune

        # Apply RandomUnstructured pruning
        prune.RandomUnstructured.apply(module, 'weight', amount)

        # Return the pruned tensor
        return module.weight
