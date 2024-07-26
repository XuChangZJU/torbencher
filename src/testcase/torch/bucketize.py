import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.bucketize)
class TorchBucketizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bucketize_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        input_tensor = torch.randn(input_size)  # Random input tensor
        num_boundaries = random.randint(1, 10)  # Random number of boundaries
        boundaries_tensor = torch.sort(
            torch.randn(num_boundaries)).values  # Generate boundaries and sort to make sure it's strictly increasing

        result = torch.bucketize(input_tensor, boundaries_tensor)
        return result
