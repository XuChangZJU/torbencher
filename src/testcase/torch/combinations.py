import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.combinations)
class TorchCombinationsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_combinations_correctness(self):
    num_elements = random.randint(3, 8)  # Random number of elements in the 1D tensor
    # print(f"Number of elements in the 1D tensor: {num_elements}")
    tensor_data = torch.randn(num_elements)  # Generate 1D tensor with random values
    r = random.randint(1, num_elements)  # Random r for combination length
    # Get combinations without replacement
    result_without_replacement = torch.combinations(tensor_data, r)
    # Get combinations with replacement
    result_with_replacement = torch.combinations(tensor_data, r, with_replacement=True)
    return result_without_replacement, result_with_replacement
