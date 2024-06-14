import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.qscheme)
class TorchTensorQschemeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_qscheme_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random tensor
        tensor = torch.randn(input_size)
        
        # Quantize the tensor
        scale = random.uniform(0.1, 1.0)
        zero_point = random.randint(0, 10)
        q_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, torch.quint8)
        
        # Get the quantization scheme
        qscheme = q_tensor.qscheme()
        
        return qscheme
    