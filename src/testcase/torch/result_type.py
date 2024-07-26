import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.result_type)
class TorchResulttypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_result_type_correctness(self):
        # Randomly generate input tensor dimensions and number of elements
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Define a list of possible torch dtypes
        dtypes = [torch.float32, torch.float64, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]
    
        # Randomly choose dtypes for tensor1 and tensor2
        dtype1 = random.choice(dtypes)
        dtype2 = random.choice(dtypes)
    
        # Create random tensors with the chosen dtypes
        tensor1 = torch.randn(input_size).to(dtype=dtype1)
        tensor2 = torch.randn(input_size).to(dtype=dtype2)
    
        # Calculate and return the result of torch.result_type
    
    
    
    
    
    
    