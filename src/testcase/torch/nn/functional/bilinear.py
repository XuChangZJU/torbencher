import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.bilinear)
class TorchNnFunctionalBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bilinear_correctness(self):
        # Random dimensions for the tensors
        batch_size = random.randint(1, 10)
        extra_dims_count = random.randint(1, 3)
        extra_dims = [random.randint(1, 5) for _ in range(extra_dims_count)]
        in1_features = random.randint(1, 10)
        in2_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
    
        # Input sizes
        input1_size = [batch_size] + extra_dims + [in1_features]
        input2_size = [batch_size] + extra_dims + [in2_features]
        weight_size = [out_features, in1_features, in2_features]
    
        # Generate random tensors
        input1 = torch.randn(input1_size)
        input2 = torch.randn(input2_size)
        weight = torch.randn(weight_size)
    
        # Calculate the result of bilinear operation
        result = torch.nn.functional.bilinear(input1, input2, weight)
        return result
    
    
    
    