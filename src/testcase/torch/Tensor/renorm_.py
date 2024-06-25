import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.renorm_)
class TorchTensorRenormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_renorm_correctness(self):
        dim = random.randint(2, 4)  # Ensure at least 2 dimensions for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor = torch.randn(input_size)
        p = random.uniform(0.1, 10.0)  # Random p value between 0.1 and 10.0
        dim = random.randint(0, len(input_size) - 1)  # Random dimension index
        maxnorm = random.uniform(0.1, 10.0)  # Random maxnorm value between 0.1 and 10.0
    
        result = tensor.renorm_(p, dim, maxnorm)
        return result
    
    
    
    