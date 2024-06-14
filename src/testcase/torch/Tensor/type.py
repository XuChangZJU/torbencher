import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.type)
class TorchTensorTypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_type_correctness(self):
        # Generate random dimension and size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random tensor
        tensor = torch.randn(input_size)
    
        # Randomly choose a dtype
        dtypes = [torch.float, torch.double, torch.int, torch.long]
        dtype = random.choice(dtypes)
    
        # Cast the tensor to the chosen dtype
        result = tensor.type(dtype)
        return result
    