import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.fuse_linear_bn_weights)
class TorchNnUtilsFuseUlinearUbnUweightsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_fuse_linear_bn_weights_correctness(self):
        # Define the dimensions for the linear and batch normalization layers
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Generate random parameters for the linear layer
        linear_weight = torch.randn(out_features, in_features)
        linear_bias = torch.randn(out_features)

        # Generate random parameters for the batch normalization layer
        bn_running_mean = torch.randn(out_features)
        bn_running_var = torch.rand(out_features)  # Should be positive
        bn_epsilon = random.uniform(1e-5, 1e-4)
        bn_weight = torch.randn(out_features)
        bn_bias = torch.randn(out_features)

        # Fuse the linear and batch normalization parameters
        fused_linear_weight, fused_linear_bias = torch.nn.utils.fuse_linear_bn_weights(
            linear_weight, linear_bias, bn_running_mean, bn_running_var, bn_epsilon, bn_weight, bn_bias
        )
        return fused_linear_weight, fused_linear_bias
