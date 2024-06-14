import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.bitwise_xor)
class TorchTensorBitwisexorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_xor_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor1 = torch.randint(0, 256, input_size, dtype=torch.int32)  # Random integer tensor1
        tensor2 = torch.randint(0, 256, input_size, dtype=torch.int32)  # Random integer tensor2
    
        result = tensor1.bitwise_xor(tensor2)
        return result
    
    
    
    