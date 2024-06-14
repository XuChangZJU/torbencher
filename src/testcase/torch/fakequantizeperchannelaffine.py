import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fakequantizeperchannelaffine)
class TorchFakequantizeperchannelaffineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fake_quantize_per_channel_affine_correctness(self):
    dim = random.randint(2, 4)  # Dimension should be larger than 2 to avoid broadcasting issue
    num_of_elements_each_dim = random.randint(1,5)
    input_size=[num_of_elements_each_dim for i in range(dim)] 
    input = torch.randn(input_size)
    channel_size = input_size[1] # channel size should match the input size
    scale = (torch.rand(channel_size) + 0.1) * 0.05 # scale should be small to show the effect of quantization
    zero_point = torch.zeros(channel_size).to(torch.int32) # zero_point should be 0 to avoid overflow
    axis = random.randint(0, dim - 1)
    quant_min = 0
    quant_max = 255
    result = torch.fake_quantize_per_channel_affine(input, scale, zero_point, axis, quant_min, quant_max)
    return result
