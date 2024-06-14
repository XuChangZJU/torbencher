import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.glu)
class TorchNnFunctionalGluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_glu_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensor
    num_of_elements_each_dim = random.randint(2, 5) * 2  # Ensure even number of elements for splitting
    input_size = [num_of_elements_each_dim for _ in range(dim)] 

    input_tensor = torch.randn(input_size)
    result = torch.nn.functional.glu(input_tensor, dim=-1)
    return result
