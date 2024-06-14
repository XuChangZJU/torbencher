import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.concat)
class TorchConcatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_concat_correctness(self):
    # Randomly generate the dimension of tensors
    dim = random.randint(1, 4)
    # Randomly generate the number of elements for each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Generate the input size for tensors
    input_size = [num_of_elements_each_dim for i in range(dim)]
    # Generate a random number of tensors to concatenate
    num_of_tensors = random.randint(2, 5)
    # Generate a list of random tensors with the same size
    tensors = [torch.randn(input_size) for i in range(num_of_tensors)]
    # Randomly select a dimension along which to concatenate the tensors
    concat_dim = random.randint(0, len(input_size) - 1) # Make sure 0 <= concat_dim < dim
    # Concatenate the tensors along the specified dimension
    result = torch.concat(tensors, concat_dim)
    return result
