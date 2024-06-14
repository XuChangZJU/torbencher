import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.initial_seed)
class TorchInitialseedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_initial_seed_correctness(self):
        # No input parameters for torch.initial_seed()
        # Generate random tensors and check if the seed affects their generation
        torch.initial_seed() # Set a random seed
    
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        tensor1 = torch.randn(input_size)
        torch.initial_seed() # Reset the seed
        tensor2 = torch.randn(input_size) 
    
        return tensor1, tensor2 # The tensors should be different due to seed reset
    
    
    
    
    
    
    