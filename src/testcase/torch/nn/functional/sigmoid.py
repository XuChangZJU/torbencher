import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.sigmoid)
class TorchNnFunctionalSigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sigmoid_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)  
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Random input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        # Generate a random tensor 
        input_tensor = torch.randn(input_size)
        # Calculate sigmoid 
        result = torch.nn.functional.sigmoid(input_tensor)
        return result
    