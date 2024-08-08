import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.MaxPool3d)
class TorchNnMaxpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_MaxPool3d_correctness(self):
        # Random input size
        dim = 3  # For 3D, we know the exact dimension count
        num_of_elements_each_dim = random.randint(4, 8)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        input_size_with_batch = [1] + input_size  # Adding batch dimension

        # Random kernel size, ensuring it fits within the input dimensions
        kernel_size_d = random.randint(1, input_size[0] - 1)
        kernel_size_h = random.randint(1, input_size[1] - 1)
        kernel_size_w = random.randint(1, input_size[2] - 1)
        kernel_size = (kernel_size_d, kernel_size_h, kernel_size_w)

        # Random stride, ensuring it's at least 1 and doesn't exceed kernel size
        stride_d = random.randint(1, kernel_size_d)
        stride_h = random.randint(1, kernel_size_h)
        stride_w = random.randint(1, kernel_size_w)
        stride = (stride_d, stride_h, stride_w)

        # 随机生成符合规则的填充大小
        half_kernel_d = kernel_size_d // 2
        half_kernel_h = kernel_size_h // 2
        half_kernel_w = kernel_size_w // 2

        padding_d = random.randint(0, half_kernel_d)  # 确保填充不超过内核大小的一半
        padding_h = random.randint(0, half_kernel_h)
        padding_w = random.randint(0, half_kernel_w)
        padding = (padding_d, padding_h, padding_w)

        # Random dilation, kept simple for clarity
        dilation_d, dilation_h, dilation_w = (1, 1, 1)  # Simplifying to avoid complexity in size calculations

        # Validate output dimensions
        # Effective kernel size considering dilation
        eff_kernel_d = (kernel_size_d - 1) * dilation_d + 1
        eff_kernel_h = (kernel_size_h - 1) * dilation_h + 1
        eff_kernel_w = (kernel_size_w - 1) * dilation_w + 1

        # Compute output dimensions
        out_d = ((input_size[0] + 2 * padding_d - eff_kernel_d) // stride_d) + 1
        out_h = ((input_size[1] + 2 * padding_h - eff_kernel_h) // stride_h) + 1
        out_w = ((input_size[2] + 2 * padding_w - eff_kernel_w) // stride_w) + 1

        assert out_d > 0 and out_h > 0 and out_w > 0, "Computed output dimensions are invalid"

        # Randomly choose whether to return indices and ceil_mode
        return_indices = random.choice([True, False])
        ceil_mode = random.choice([True, False])

        # Create the input tensor
        input_tensor = torch.randn(input_size_with_batch)

        # Create the MaxPool3d layer with validated parameters
        max_pool_3d = torch.nn.MaxPool3d(kernel_size, stride, padding, dilation=(dilation_d, dilation_h, dilation_w),
                                         return_indices=return_indices, ceil_mode=ceil_mode)

        # Apply the MaxPool3d layer
        result = max_pool_3d(input_tensor)

        return result
