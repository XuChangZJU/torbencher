import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cond)
class TorchCondTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cond_correctness(self):
        # Define two functions that will be used in the conditional
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        # Randomly decide which branch to take (either boolean or tensor with a single boolean element)
        if random.choice([True, False]):
            pred = random.choice([True, False])
        else:
            pred = torch.tensor(random.choice([True, False]))

        # Generate random tensor size and initialize the operand tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        operand = torch.randn(input_size)

        # Apply `torch.cond` and return the result
        result = torch.cond(pred, true_fn, false_fn, (operand,))
        return result
