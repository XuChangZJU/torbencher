import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.lessequal)
class TorchLessequalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_less_equal_correctness(self):
        # Generate random dimension and size for input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random tensors of the same size
        input1 = torch.randn(input_size)
        input2 = torch.randn(input_size)
    
        # Calculate the less_equal result
        result = torch.less_equal(input1, input2)
        
        return result
    