import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.FractionalMaxPool3d)
class TorchNnFractionalmaxpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fractional_max_pool3d_correctness(self):
        # 随机生成输入张量的尺寸
        batch_size = random.randint(1, 5)  # 随机批量大小
        channels = random.randint(1, 5)  # 随机通道数
        depth = random.randint(10, 20)  # 随机深度
        height = random.randint(10, 20)  # 随机高度
        width = random.randint(10, 20)  # 随机宽度

        # 生成具有指定尺寸的随机输入张量
        input_tensor = torch.randn(batch_size, channels, depth, height, width)

        # 随机生成池化核大小，确保池化核尺寸不超过输入尺寸的一半以避免潜在错误
        kernel_size = random.randint(2, min(5, depth // 2, height // 2, width // 2))

        # 随机决定使用输出尺寸还是输出比例
        if random.choice([True, False]):
            # 使用输出尺寸
            # 确保输出尺寸小于输入尺寸
            output_depth = random.randint(1, depth // 2)  # 输出深度不超过输入深度的一半
            output_height = random.randint(1, height // 2)  # 输出高度不超过输入高度的一半
            output_width = random.randint(1, width // 2)  # 输出宽度不超过输入宽度的一半
            pool_layer = torch.nn.FractionalMaxPool3d(kernel_size,
                                                      output_size=(output_depth, output_height, output_width))
        else:
            # 使用输出比例，确保比例乘以输入尺寸后为整数且不超过原尺寸的一半
            output_ratio_depth = random.uniform(0.1, 0.5)  # 比例限制在0.1到0.5之间以避免尺寸过大
            output_ratio_height = random.uniform(0.1, 0.5)
            output_ratio_width = random.uniform(0.1, 0.5)

            # 直接计算实际的输出尺寸，不再需要min函数限制，因为我们已经限制了比例
            output_depth = int(depth * output_ratio_depth)
            output_height = int(height * output_ratio_height)
            output_width = int(width * output_ratio_width)

            pool_layer = torch.nn.FractionalMaxPool3d(kernel_size, output_ratio=(
            output_ratio_depth, output_ratio_height, output_ratio_width))

        # 应用分数最大池化操作
        output_tensor = pool_layer(input_tensor)

        return output_tensor
