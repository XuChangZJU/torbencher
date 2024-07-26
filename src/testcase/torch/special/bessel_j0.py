import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.bessel_j0)
class TorchSpecialBesselj0TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_special_bessel_j0_correctness(self):
        # Define the dimension of the tensor
        dim = random.randint(1, 4)
        # Define the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor of the specified size
        input_tensor = torch.randn(input_size)
        # Calculate the Bessel function of the first kind of order 0
        result = torch.special.bessel_j0(input_tensor)
        # Return the result
        return result
    