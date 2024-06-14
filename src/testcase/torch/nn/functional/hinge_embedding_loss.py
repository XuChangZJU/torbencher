import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hinge_embedding_loss)
class TorchNnFunctionalHingeembeddinglossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hinge_embedding_loss_correctness(self):
        # Define the dimensions for the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensors
        input = torch.randn(input_size)
        target = torch.randn(input_size)  # Target tensor should have the same size as input
    
        # Calculate the hinge embedding loss
        loss = torch.nn.functional.hinge_embedding_loss(input, target)
    
        return loss
    
    
    
    