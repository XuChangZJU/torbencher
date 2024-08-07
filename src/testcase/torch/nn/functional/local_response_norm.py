import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.local_response_norm)
class TorchNnFunctionalLocalUresponseUnormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_local_response_norm_correctness(self):
        # Define the input size
        num_of_dimensions = random.randint(3, 5)
        num_of_elements_each_dim = random.randint(1, 3)
        input_size = [num_of_elements_each_dim for i in range(num_of_dimensions)]

        # Generate a random input tensor 
        input_tensor = torch.randn(input_size)

        # Define the parameters for local response normalization
        size = random.randint(1, input_size[1])  # size should be less than or equal to the number of channels

        # Apply local response normalization
        output_tensor = torch.nn.functional.local_response_norm(input_tensor, size)

        return output_tensor
