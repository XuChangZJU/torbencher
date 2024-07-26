import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.unravel_index)
class TorchUnravelindexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unravel_index_correctness(self):
        # Randomly generate the dimension of the shape
        dim = random.randint(1, 4)
        # Randomly generate the number of elements for each dimension
        num_of_elements_each_dim = [random.randint(1, 5) for i in range(dim)]
        # Generate the shape of the tensor
        shape = tuple(num_of_elements_each_dim)
        # Calculate the total number of elements in the tensor
        total_elements = torch.prod(torch.tensor(shape)).item()
        # Generate random indices within the valid range
        indices = torch.randint(0, total_elements, (random.randint(1, 5), random.randint(1, 5)))
        # Call the unravel_index function
        result = torch.unravel_index(indices, shape)
        return result
    
    
    
    
    
    
    