import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.fuseconvbnweights)
class TorchNnUtilsFuseconvbnweightsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fuse_conv_bn_weights_correctness(self):
        # Define the dimensions for the convolutional layer
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 5)
    
        # Generate random parameters for the convolutional layer
        conv_w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        conv_b = torch.randn(out_channels)
    
        # Generate random parameters for the batch normalization layer
        bn_rm = torch.randn(out_channels)
        bn_rv = torch.rand(out_channels)  # Ensure running variance is positive
        bn_eps = 1e-5
        bn_w = torch.randn(out_channels)
        bn_b = torch.randn(out_channels)
    
        # Fuse the convolutional and batch normalization parameters
        fused_conv_w, fused_conv_b = torch.nn.utils.fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b)
    
        return fused_conv_w, fused_conv_b
    