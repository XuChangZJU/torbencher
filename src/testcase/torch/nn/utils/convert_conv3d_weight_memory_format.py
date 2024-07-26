import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.convert_conv3d_weight_memory_format)
class TorchNnUtilsConvertconv3dweightmemoryformatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_convert_conv3d_weight_memory_format_correctness(self):
        # Random input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random Conv3d parameters
        in_channels = random.randint(1, 10)
        out_channels = random.randint(1, 10)
        kernel_size = random.randint(1, 5)

        # Create a Conv3d module
        conv3d_module = torch.nn.Conv3d(in_channels, out_channels, kernel_size)

        # Convert memory format to channels_last
        memory_format = torch.channels_last
        result = torch.nn.utils.convert_conv3d_weight_memory_format(conv3d_module, memory_format)

        return result
