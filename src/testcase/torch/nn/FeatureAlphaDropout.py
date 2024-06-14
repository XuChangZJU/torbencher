import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.FeatureAlphaDropout)
class TorchNnFeaturealphadropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_feature_alpha_dropout_correctness(self):
        # Randomly generate dimensions for the input tensor
        dim = random.randint(3, 5)  # Random dimension for the tensors (3 to 5 dimensions)
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random tensor with the specified dimensions
        input_tensor = torch.randn(input_size)
    
        # Randomly generate the dropout probability
        p = random.uniform(0.1, 0.9)  # Dropout probability between 0.1 and 0.9
    
        # Create the FeatureAlphaDropout module with the generated probability
        feature_alpha_dropout = torch.nn.FeatureAlphaDropout(p)
    
        # Apply the dropout to the input tensor
        output_tensor = feature_alpha_dropout(input_tensor)
    
        return output_tensor
    