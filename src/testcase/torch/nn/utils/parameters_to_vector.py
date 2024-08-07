import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.parameters_to_vector)
class TorchNnUtilsParametersUtoUvectorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_parameters_to_vector_correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size based on dim and num_of_elements_each_dim
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a list of random tensors with the same input_size
        parameters = [torch.randn(input_size) for i in range(random.randint(1, 5))]
        # Flatten parameters into a single vector
        result = torch.nn.utils.parameters_to_vector(parameters)
        return result
