import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.sumtosize)
class TorchTensorSumtosizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sum_to_size_correctness(self):
    # Randomly generate tensor dimension and size
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random tensor 
    tensor = torch.randn(input_size)

    # Generate random size to which the tensor will be summed
    # size should be broadcastable to tensor's size, 
    # meaning its each dimension should be either equal to the corresponding dimension in tensor's size
    # or equal to 1.
    size = []
    for i in range(dim):
        if random.random() < 0.5:
            size.append(tensor.size(i))
        else:
            size.append(1)

    result = tensor.sum_to_size(size)
    return result
