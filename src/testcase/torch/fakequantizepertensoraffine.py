import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fakequantizepertensoraffine)
class TorchFakequantizepertensoraffineTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fake_quantize_per_tensor_affine_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Generate a random tensor with the specified size
        input_tensor = torch.randn(input_size, dtype=torch.float32)
        
        # Random scale value between 0.1 and 1.0
        scale = random.uniform(0.1, 1.0)
        
        # Random zero_point value between -128 and 127
        zero_point = random.randint(-128, 127)
        
        # Bounds for quantization
        quant_min = random.randint(-128, -127)
        quant_max = random.randint(126, 127)
        
        # Ensure that quant_min is less than quant_max
        if quant_min >= quant_max:
            quant_min, quant_max = quant_max, quant_min
        
        # Perform the fake quantization
        result = torch.fake_quantize_per_tensor_affine(input_tensor, 
                                                       scale, 
                                                       zero_point, 
                                                       quant_min, 
                                                       quant_max)
        
        return result
    