import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.conj_physical)
class TorchConjUphysicalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conj_physical_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create a list representing the size of the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Create a random tensor of complex data type
        complex_tensor = torch.randn(input_size) + torch.randn(input_size) * 1j
        complex_result = torch.conj_physical(complex_tensor)

        # Create a random tensor of non-complex data type
        non_complex_tensor = torch.randn(input_size)
        non_complex_result = torch.conj_physical(non_complex_tensor)

        return complex_result, non_complex_result
