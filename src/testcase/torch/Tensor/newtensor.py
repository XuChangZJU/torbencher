import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.newtensor)
class TorchTensorNewtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_new_tensor_correctness(self):
    dim = random.randint(1, 4)  # Random dimension for the tensors
    num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
    input_size=[num_of_elements_each_dim for i in range(dim)] 

    original_tensor = torch.randn(input_size) # Random original tensor
    data = [[random.uniform(0, 1) for _ in range(len(input_size))] for _ in range(len(input_size))] # Random data for the new tensor
    result = original_tensor.new_tensor(data)
    return result
