import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.geometric.Geometric)
class TorchDistributionsGeometricGeometricTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_geometric_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random probabilities in the range (0, 1]
        probs = torch.rand(input_size) * 0.99 + 0.01
        # Create a Geometric distribution
        m = torch.distributions.geometric.Geometric(probs)
        # Sample from the distribution
        result = m.sample()
        return result
    