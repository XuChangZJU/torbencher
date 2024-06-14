import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dstack)
class TorchDstackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dstack_correctness(self):
    # Randomly generate the dimension of input tensors, which should be less than or equal to 3
    dim = random.randint(1, 3)
    # Randomly generate the number of elements for each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Generate the input size list for tensors
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Generate a random number of tensors to be stacked
    num_of_tensors = random.randint(2, 5)
    # Generate a list of random tensors
    tensors = [torch.randn(input_size) for _ in range(num_of_tensors)]
    # Apply dstack operation
    result = torch.dstack(tensors)
    return result
