import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_leaf)
class TorchTensorIsUleafTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_leaf_correctness(self):
        # 使用随机数生成张量
        a = torch.randn(3, requires_grad=True)
        result_a = a.is_leaf

        # 通过操作生成张量
        b = a + 2
        result_b = b.is_leaf

        # 通过操作生成张量并设置requires_grad=True
        c = b * 2
        result_c = c.is_leaf

        # 使用随机数生成不需要梯度的张量
        d = torch.randn(3)
        result_d = d.is_leaf

        # 设置requires_grad=True
        e = d.clone().detach().requires_grad_(True)
        result_e = e.is_leaf

        # 通过操作生成张量后设置requires_grad=True
        f = e + 3
        f = f.requires_grad_(True)
        result_f = f.is_leaf
        return result_a, result_b, result_c, result_d, result_e, result_f
