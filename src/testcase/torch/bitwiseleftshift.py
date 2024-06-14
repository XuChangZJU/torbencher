import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwiseleftshift)
class TorchBitwiseleftshiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_left_shift_correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors with integer type
        input_tensor = torch.randint(-10, 10, input_size) # Generate random integers between -10 and 10
        shift_amount_tensor = torch.randint(0, 8, input_size) # Generate random shift amounts between 0 and 7
    
        # Perform bitwise left shift operation
        result = torch.bitwise_left_shift(input_tensor, shift_amount_tensor)
        return result
    