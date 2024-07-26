import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AlphaDropout)
class TorchNnAlphadropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_AlphaDropout_correctness(self):
        # Randomly generate input tensor dimension
        dim = random.randint(1, 4)
        # Randomly generate number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input_size
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Randomly generate p
        p = random.uniform(0, 1)
        # Define AlphaDropout module
        alpha_dropout = torch.nn.AlphaDropout(p)
    
        # Test training mode
        alpha_dropout.train()
        output_tensor_train = alpha_dropout(input_tensor)
    
        # Test evaluation mode
        alpha_dropout.eval()
        output_tensor_eval = alpha_dropout(input_tensor)
    
        return output_tensor_train, output_tensor_eval
    
    
    
    