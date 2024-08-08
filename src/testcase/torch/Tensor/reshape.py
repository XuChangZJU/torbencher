import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.reshape)
class TorchTensorReshapeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_reshape_correctness(self):
        # 随机生成原始张量的维度
        original_dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        original_size = [num_of_elements_each_dim] * original_dim

        # 创建具有生成尺寸的随机张量
        original_tensor = torch.randn(original_size)

        # 随机生成与元素总数兼容的新形状
        perm = torch.randperm(len(original_size))
        new_shape = [original_size[i] for i in perm]

        # 将张量重塑为新形状
        reshaped_tensor = original_tensor.reshape(new_shape)

        return reshaped_tensor
