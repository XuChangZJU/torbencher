import random
import torch
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize the weights using the orthogonal_ method
        torch.nn.init.orthogonal_(self.linear.weight.data)

@test_api(torch.nn.init.orthogonal_)
class TorchNnInitOrthogonalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_orthogonal__correctness(self):
        # Randomly generate the input and output dimensions
        in_features = random.randint(1, 5)
        out_features = random.randint(1, 5)
        
        # Initialize the model
        model = SimpleLinearModel(in_features, out_features)
        
        # Optionally, verify that the weights have been initialized correctly
        # For example, you might check the condition number or orthogonality of the weight matrix
        # Here we simply return the weight matrix to demonstrate the initialization
        return model.linear.weight.data
