import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.renorm)
class TorchRenormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_renorm_correctness(self):
        # Randomly select the dimensions for the tensor
        dim = random.randint(2, 4)  # Minimum dimension should be 2 to ensure slicing over one dimension
        num_of_elements_each_dim = random.randint(1, 5)  # Random element count for each dimension
        
        # Construct the input size
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Randomly generate the input tensor
        input_tensor = torch.randn(input_size)
        
        # Generate the p value for norm computation, must be > 0 (let's use range 0.1 to 10 for practical purposes)
        p = random.uniform(0.1, 10.0)
        
        # Randomly select the dimension to slice and normalize over
        norm_dim = random.randint(0, dim - 1)  # Ensure it's a valid dimension index
        
        # Randomly select the max norm value to keep each sub-tensor under
        maxnorm = random.uniform(0.1, 10.0)
        
        # Perform the renorm operation
        result = torch.renorm(input_tensor, p, norm_dim, maxnorm)
        
        # Return the result tensor
        return result
    