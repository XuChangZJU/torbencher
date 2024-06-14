import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.bitwise_right_shift_)
class TorchTensorBitwiserightshiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift__correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4) 
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Generate input_size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate random tensor
        self = torch.randint(0, 10, input_size)
        # Generate other tensor with the same size as self
        other = torch.randint(0, 10, input_size)
        # Perform the operation
        result = self.bitwise_right_shift_(other)
        # Return the result
        return result
    
    
    
    