import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_printoptions)
class TorchSetprintoptionsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_printoptions_correctness(self):
        # Randomly generate parameters for torch.set_printoptions
        precision = random.randint(0, 10)  # Random precision between 0 and 10
        threshold = random.randint(1, 1000)  # Random threshold between 1 and 1000
        edgeitems = random.randint(1, 10)  # Random edgeitems between 1 and 10
        linewidth = random.randint(10, 100)  # Random linewidth between 10 and 100
    
        # Generate a random tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        tensor = torch.randn(input_size)
    
        # Set print options
        torch.set_printoptions(precision, threshold, edgeitems, linewidth)
    
        # Trigger the effect of torch.set_printoptions
        result = tensor.__repr__()
    
        return result
    
    
    
    
    
    
    