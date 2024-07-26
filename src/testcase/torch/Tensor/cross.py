import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.cross)
class TorchTensorCrossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross_correctness(self):
        dim = random.randint(3, 5)
        num_of_elements_each_dim = 3  # 被叉乘的那个维度的大小必须为3
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        cross_dim = random.randint(0, dim - 1)  # Random dimension along which to compute the cross product

        result = tensor1.cross(tensor2, cross_dim)
        return result
