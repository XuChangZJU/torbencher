import torch
import random
from torch.nn.utils import prune


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.globalunstructured)
class TorchNnUtilsPruneGlobalunstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_global_unstructured_correctness(self):
        # Define the size of the neural network
        input_size = random.randint(1, 10)
        hidden_size = random.randint(1, 10)
        output_size = random.randint(1, 10)
    
        # Create a simple neural network
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
    