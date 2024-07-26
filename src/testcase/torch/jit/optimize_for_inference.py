import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.optimize_for_inference)
class TorchJitOptimizeforinferenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_optimize_for_inference_correctness(self):
        # Define input size randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Define random input tensor
        input_tensor = torch.randn(input_size)

        # Define a simple model (Conv->BatchNorm)
        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)
        bn = torch.nn.BatchNorm2d(out_channels, eps=.001)
        mod = torch.nn.Sequential(conv, bn)

        # Optimize the model for inference
        optimized_mod = torch.jit.optimize_for_inference(torch.jit.script(mod.eval()))

        # Return the optimized model
        return optimized_mod
