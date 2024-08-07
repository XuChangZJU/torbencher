import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.special.spherical_bessel_j0)
class TorchSpecialSphericalUbesselUj0TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_spherical_bessel_j0_correctness(self):
        # Define the dimension of the input tensor randomly.
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension randomly.
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate the input size.
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random input tensor.
        input_tensor = torch.randn(input_size)
        # Calculate the spherical Bessel function of the first kind of order 0.
        result = torch.special.spherical_bessel_j0(input_tensor)
        # Return the result.
        return result
