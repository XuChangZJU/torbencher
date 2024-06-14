import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.ismeta)
class TorchTensorIsmetaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_meta_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    tensor = torch.randn(input_size) # Generate a normal tensor
    result1 = tensor.is_meta # Check if the normal tensor is a meta tensor

    tensor = torch.randn(input_size).to('meta') # Generate a meta tensor
    result2 = tensor.is_meta # Check if the meta tensor is a meta tensor
    
    return result1, result2
