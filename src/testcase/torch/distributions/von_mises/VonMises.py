import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.von_mises.VonMises)
class TorchDistributionsVonmisesVonmisesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_VonMises_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random loc tensor
        loc = torch.randn(input_size)
        # Random concentration tensor, concentration > 0
        concentration = torch.rand(input_size) + 1e-5
        # Create a VonMises distribution
        von_mises_distribution = torch.distributions.von_mises.VonMises(loc, concentration)
        # Sample from the distribution
        result = von_mises_distribution.sample()
        return result
