import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.bitwiserightshift)
class TorchTensorBitwiserightshiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_correctness(self):
        # Define the dimension and size of the tensors randomly
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors
        input_tensor = torch.randint(0, 10, input_size) # Generate integers to clearly see the effect of bitwise shift
        shift_amount = random.randint(0, 5) # Generate random shift amount
    
        # Perform bitwise right shift operation
        result = input_tensor.bitwise_right_shift(shift_amount)
        
        return result
    