import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.multilabel_margin_loss)
class TorchNnFunctionalMultilabelmarginlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multilabel_margin_loss_correctness(self):
        # Define the dimensions for the input tensor
        dim1 = random.randint(1, 10)  # Batch size
        dim2 = random.randint(1, 10)  # Number of classes
    
        # Generate random input tensor
        input_tensor = torch.randn(dim1, dim2)
    
        # Generate random target tensor with values in the range [-1, dim2 - 1]
        target_tensor = torch.randint(-1, dim2, (dim1, dim2))
    
        # Calculate the multi-label margin loss
        loss = torch.nn.functional.multilabel_margin_loss(input_tensor, target_tensor)
        
        return loss
    
    
    
    