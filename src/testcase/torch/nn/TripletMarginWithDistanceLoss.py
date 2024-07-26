import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.TripletMarginWithDistanceLoss)
class TorchNnTripletmarginwithdistancelossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_TripletMarginWithDistanceLoss_correctness(self):
        # Define the dimension and size of the input tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensors
        anchor = torch.randn(input_size)
        positive = torch.randn(input_size)
        negative = torch.randn(input_size)
    
        # Create a TripletMarginWithDistanceLoss object
        triplet_loss = torch.nn.TripletMarginWithDistanceLoss()
    
        # Calculate the loss
        loss = triplet_loss(anchor, positive, negative)
    
        # Return the loss
        return loss
    
    
    