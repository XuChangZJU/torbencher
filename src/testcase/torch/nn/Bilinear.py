import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Bilinear)
class TorchNnBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bilinear_correctness(self):
        # Randomly generate the size of each input sample
        in1_features = random.randint(1, 10)
        in2_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        
        # Create a Bilinear layer with the generated sizes
        bilinear_layer = torch.nn.Bilinear(in1_features, in2_features, out_features)
        
        # Randomly generate the batch size
        batch_size = random.randint(1, 5)
        
        # Generate random input tensors with the appropriate sizes
        input1 = torch.randn(batch_size, in1_features)
        input2 = torch.randn(batch_size, in2_features)
        
        # Apply the bilinear transformation
        output = bilinear_layer(input1, input2)
        
        return output
    
    
    
    