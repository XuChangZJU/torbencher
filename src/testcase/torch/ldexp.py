import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.ldexp)
class TorchLdexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ldexp_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    mantissa = torch.randn(input_size)  # Random mantissa values
    exponents = torch.randint(-5, 5, input_size)  # Random exponents between -5 and 5
    result = torch.ldexp(mantissa, exponents)
    return result
