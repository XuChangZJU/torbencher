import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.from_numpy)
class TorchFromnumpyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_from_numpy_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the array
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        numpy_array = torch.randn(input_size).numpy() # Generate a random numpy array
        tensor = torch.from_numpy(numpy_array) # Create a tensor from the numpy array
        return tensor
    
    
    
    
    
    
    