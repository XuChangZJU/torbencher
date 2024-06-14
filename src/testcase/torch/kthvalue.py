import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.kthvalue)
class TorchKthvalueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kthvalue_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(4, 10)  # Ensure there are multiple values to find the k-th smallest
    input_size = [num_of_elements_each_dim for _ in range(dim)]

    input_tensor = torch.randn(input_size)
    k = random.randint(1, num_of_elements_each_dim)  # k should be within the number of elements in chosen dimension
    chosen_dim = random.randint(0, dim - 1)  # Random valid dimension index

    values, indices = torch.kthvalue(input_tensor, k, chosen_dim)
    return values, indices
