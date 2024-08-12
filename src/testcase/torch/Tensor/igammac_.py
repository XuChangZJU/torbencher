import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.igammac_)
class TorchTensorIgammacUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_igammac__correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor1
        tensor1 = torch.abs(torch.randn(input_size)) + 1e-5
        # Generate random tensor2, make sure tensor1 > tensor2
        tensor2 =  torch.abs(torch.randn(input_size))
        # tensor1 = torch.where(tensor1 > tensor2, tensor1, tensor2 + abs(tensor1) + 1)
        tensor1.igammac_(tensor2)
        return tensor1
