import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.fuse_conv_bn_eval)
class TorchNnUtilsFuseUconvUbnUevalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fuse_conv_bn_eval_correctness(self):
        # Randomly generate dimensions for convolutional layer
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 5)
        stride = random.randint(1, 3)
        padding = random.randint(0, 2)

        # Create convolutional layer with random parameters
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # Create BatchNorm layer with matching out_channels
        bn = torch.nn.BatchNorm2d(out_channels)

        # Set both layers to evaluation mode
        conv.eval()
        bn.eval()

        # Fuse the convolutional and batch normalization layers
        fused_conv = torch.nn.utils.fuse_conv_bn_eval(conv, bn)

        # Generate random input tensor with appropriate dimensions
        input_tensor = torch.randn(1, in_channels, random.randint(10, 20), random.randint(10, 20))

        # Pass the input tensor through the original and fused layers
        original_output = conv(input_tensor)
        fused_output = fused_conv(input_tensor)

        return original_output, fused_output
