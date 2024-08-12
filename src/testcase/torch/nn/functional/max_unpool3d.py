import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.max_unpool3d)
class TorchNnFunctionalMaxUunpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_max_unpool3d_correctness(self):
        # 随机生成输入张量的维度
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        depth = random.randint(8, 16)  # 为了保证 pooling 后的张量仍有一定的大小
        height = random.randint(8, 16)
        width = random.randint(8, 16)

        # 生成随机输入张量
        input_tensor = torch.randn(batch_size, channels, depth, height, width)

        # 定义 max pooling 的 kernel size, stride 和 padding
        kernel_size = random.randint(2, 4)
        stride = kernel_size  # 为确保 unpooling 的有效性，stride 应与 kernel_size 相同
        padding = 0  # 为了简单起见，无 padding

        # 执行 max_pool3d 操作
        pooled, indices = torch.nn.functional.max_pool3d(input_tensor, kernel_size, stride, padding,
                                                         return_indices=True)

        # 计算 unpooling 后的输出尺寸 (应该是原始输入的尺寸)
        output_size = input_tensor.size()

        # 执行 max_unpool3d 操作
        unpooled = torch.nn.functional.max_unpool3d(pooled, indices, kernel_size, stride, padding, output_size)

        return pooled, unpooled
