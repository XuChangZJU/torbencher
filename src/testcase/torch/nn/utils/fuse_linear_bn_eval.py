import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.fuse_linear_bn_eval)
class TorchNnUtilsFuselinearbnevalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fuse_linear_bn_eval_correctness(self):
        # Randomly generate input size for the linear layer
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)

        # Create a random linear layer
        linear = torch.nn.Linear(in_features, out_features)

        # Create a random batch normalization layer
        bn = torch.nn.BatchNorm1d(out_features)

        # Set both layers to evaluation mode
        linear.eval()
        bn.eval()

        # Generate random input tensor
        input_tensor = torch.randn(random.randint(1, 5), in_features)

        # Compute the output of the original linear and batch norm layers
        original_output = bn(linear(input_tensor))

        # Fuse the linear and batch norm layers
        fused_linear = torch.nn.utils.fuse_linear_bn_eval(linear, bn)

        # Compute the output of the fused layer
        fused_output = fused_linear(input_tensor)

        # Return the original and fused outputs for comparison
        return original_output, fused_output
