import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.gumbel_softmax)
class TorchNnFunctionalGumbelUsoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gumbel_softmax_correctness(self):
        # Randomly generate input tensor dimension
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random input tensor
        logits = torch.randn(input_size)
        # Generate random tau value
        tau = random.uniform(0.1, 10.0)
        # Randomly generate hard value
        hard = random.choice([True, False])
        # Calculate gumbel_softmax
        result = torch.nn.functional.gumbel_softmax(logits, tau, hard)
        return result
