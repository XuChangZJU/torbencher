import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addcdiv)
class TorchAddcdivTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addcdiv_correctness(self):
        # Define the dimension and size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors with the specified size
        input_tensor = torch.randn(input_size)
        tensor1 = torch.randn(input_size)
        # Generate random tensor2 and make sure the elements are not zero
        tensor2 = torch.randn(input_size)
        tensor2[tensor2 == 0] = 1e-6
        result = torch.addcdiv(input_tensor, tensor1, tensor2)
        return result
    