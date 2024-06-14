import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.l1loss)
class TorchNnFunctionalL1lossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_l1_loss_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input = torch.randn(input_size)
        target = torch.randn(input_size) 
        result = torch.nn.functional.l1_loss(input, target)
        return result
    