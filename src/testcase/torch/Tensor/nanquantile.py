import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.nanquantile)
class TorchTensorNanquantileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nanquantile_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random tensor 
        tensor = torch.randn(input_size)
        # Random q between 0 and 1
        q = random.uniform(0, 1)
        # Randomly select a dimension
        dim = random.randint(0, dim - 1)
        result = tensor.nanquantile(q, dim)
        return result
    
    
    
    