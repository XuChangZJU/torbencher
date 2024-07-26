import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.categorical.Categorical)
class TorchDistributionsCategoricalCategoricalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_Categorical_correctness(self):
        # Create random dimension for the probabilities tensor.
        dim = random.randint(1, 4)
        # Create random number of elements for each dimension.
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input size for the probabilities tensor.
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random probabilities tensor.
        probs = torch.rand(input_size)
        # Normalize the probabilities tensor to sum to 1 along the last dimension.
        probs = probs / probs.sum(dim=-1, keepdim=True)
        # Create Categorical distribution.
        m = torch.distributions.categorical.Categorical(probs)
        # Sample from the distribution.
        result = m.sample()
        return result
