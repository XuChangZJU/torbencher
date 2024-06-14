import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.dequantize)
class TorchTensorDequantizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dequantize_correctness(self):
        # Randomly generate dimensions for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random quantized tensor
        scale = random.uniform(0.1, 1.0)  # Random scale between 0.1 and 1.0
        zero_point = random.randint(0, 255)  # Random zero_point between 0 and 255
        quantized_tensor = torch.quantize_per_tensor(torch.randn(input_size), scale, zero_point, torch.quint8)
    
        # Dequantize the tensor
        dequantized_tensor = quantized_tensor.dequantize()
        return dequantized_tensor
    