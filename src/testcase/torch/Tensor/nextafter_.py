import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.nextafter_)
class TorchTensorNextafterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nextafter__correctness(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)  
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Generate input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate random tensor1
        tensor1 = torch.randn(input_size)
        # Generate random tensor2
        tensor2 = torch.randn(input_size)
        # Call nextafter_
        tensor1.nextafter_(tensor2)
        # Return tensor1 to check the effect of nextafter_
        return tensor1
    
    
    
    