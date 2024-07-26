import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.q_zero_point)
class TorchTensorQzeropointTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_q_zero_point_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate random tensor data
        tensor = torch.randn(input_size)
    
        # Generate random quantization parameters
        scale = random.uniform(0.1, 10.0)  # Random scale value between 0.1 and 10.0
        zero_point = random.randint(-128, 127)  # Random zero_point value between -128 and 127
    
        # Quantize the tensor
        quantized_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, torch.qint8)
    
        # Get the zero_point of the quantized tensor
        result = quantized_tensor.q_zero_point()
        return result
    
    
    
    