import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.kldiv)
class TorchNnFunctionalKldivTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kl_div_correctness(self):
    # Randomly generate the dimension of the input tensors
    dim = random.randint(1, 4)
    # Randomly generate the number of elements for each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Create a list representing the size of the input tensors
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random input tensor in log-probabilities
    input = torch.randn(input_size)
    input = torch.nn.functional.log_softmax(input, dim=random.randint(0, dim - 1)) # Ensure valid probabilities

    # Generate random target tensor with the same size as input
    target = torch.randn(input_size)
    target = torch.nn.functional.softmax(target, dim=random.randint(0, dim - 1)) # Ensure valid probabilities

    # Calculate KL Divergence loss with different reduction methods
    result_none = torch.nn.functional.kl_div(input, target, reduction='none')
    result_batchmean = torch.nn.functional.kl_div(input, target, reduction='batchmean')
    result_sum = torch.nn.functional.kl_div(input, target, reduction='sum')
    result_mean = torch.nn.functional.kl_div(input, target, reduction='mean')

    return result_none, result_batchmean, result_sum, result_mean
