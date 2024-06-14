import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.iscomplex)
class TorchIscomplexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_complex_correctness(self):
        # Create random dimension and number of elements for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Create a random tensor of complex data type (torch.complex64 or torch.complex128)
        tensor = torch.randn(input_size, dtype=random.choice([torch.complex64, torch.complex128]))
        
        result = torch.is_complex(tensor)
        return result
    