import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwiserightshift)
class TorchBitwiserightshiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_correctness(self):
        # Define the dimension and size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors with int type
        input_tensor = torch.randint(-100, 100, input_size)  # Generate random integers between -100 and 100
        shift_amount = torch.randint(0, 8, input_size) # Generate random shift amounts between 0 and 7 (valid for int8)
    
        result = torch.bitwise_right_shift(input_tensor, shift_amount)
        return result
    