import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.resizeas)
class TorchTensorResizeasTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_resize_as_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        target_size = [random.randint(1, 10) for _ in range(dim)]  # Generate random target size
        target_tensor = torch.randn(target_size)
        
        result = input_tensor.resize_as_(target_tensor)
        return result
    