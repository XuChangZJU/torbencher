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
        # Initialize the weights using the uniform_ method
        a = random.uniform(-10.0, 10.0)
        b = random.uniform(a, 10.0)  # Ensure b is greater than a
        torch.nn.init.uniform_(self.linear.weight.data, a, b)


@test_api(torch.nn.init.uniform_)
class TorchNnInitOrthogonalUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_uniform_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        
        # Random input and output features for the linear layer
        in_features = random.randint(1, 5)
        out_features = random.randint(1, 5)
        
        # Initialize the model
        model = SimpleLinearModel(in_features, out_features)
        
        # Return the initialized weight matrix for verification
        return model.linear.weight.data
