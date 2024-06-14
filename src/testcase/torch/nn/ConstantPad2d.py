import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ConstantPad2d)
class TorchNnConstantpad2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ConstantPad2d_correctness(self):
        # Random input size
        dim = random.randint(2, 4)  # Dimension should be at least 2 for ConstantPad2d
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Random padding
        padding_int = random.randint(1, 3)  # Random integer padding
        padding_tuple = (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))  # Random tuple padding
    
        # Random input tensor
        input_tensor = torch.randn(input_size)
    
        # Test with integer padding
        constant_pad_int = torch.nn.ConstantPad2d(padding_int, 3.5)
        result_int_padding = constant_pad_int(input_tensor)
    
        # Test with tuple padding
        constant_pad_tuple = torch.nn.ConstantPad2d(padding_tuple, 3.5)
        result_tuple_padding = constant_pad_tuple(input_tensor)
    
        return result_int_padding, result_tuple_padding
    