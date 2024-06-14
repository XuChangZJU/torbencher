import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.binarycrossentropy)
class TorchNnFunctionalBinarycrossentropyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_correctness(self):
        """
        Test the correctness of binary_cross_entropy with random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor with values between 0 and 1
        input_tensor = torch.rand(input_size) 
        # Generate random target tensor with values between 0 and 1
        target_tensor = torch.rand(input_size) 
        
        # Calculate binary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy(input_tensor, target_tensor)
        return loss
    