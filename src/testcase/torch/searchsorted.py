import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.searchsorted)
class TorchSearchsortedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_searchsorted_correctness(self):
    # Define the dimension and size of the input tensors
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate a sorted tensor for sorted_sequence
    sorted_sequence = torch.sort(torch.randn(input_size)).values

    # Generate a tensor for values
    values = torch.randn(input_size)

    # Call searchsorted
    result = torch.searchsorted(sorted_sequence, values)
    return result
