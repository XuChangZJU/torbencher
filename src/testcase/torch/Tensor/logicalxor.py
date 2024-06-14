import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.logicalxor)
class TorchTensorLogicalxorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_xor_correctness(self):
    # Generate random dimension and size for input tensors
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random tensors with values 0 or 1
    tensor1 = torch.randint(0, 2, input_size) # Only 0 and 1 are valid for logical_xor
    tensor2 = torch.randint(0, 2, input_size) # Only 0 and 1 are valid for logical_xor
    
    # Perform logical XOR operation
    result = tensor1.logical_xor(tensor2)
    
    return result
