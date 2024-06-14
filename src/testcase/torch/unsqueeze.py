import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.unsqueeze)
class TorchUnsqueezeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unsqueeze_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 
    input_tensor = torch.randn(input_size)
    dim_to_unsqueeze = random.randint(-dim-1, dim) # Random valid dim to unsqueeze
    result = torch.unsqueeze(input_tensor, dim_to_unsqueeze)
    return result
